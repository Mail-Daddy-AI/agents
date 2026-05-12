from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, Literal

from livekit import rtc

from .. import utils
from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio
from ..utils.audio import AudioBuffer
from ..vad import VAD
from .stt import STT, RecognizeStream, SpeechEvent, SpeechEventType, STTCapabilities

# don't retry when using the fallback adapter
DEFAULT_FALLBACK_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class AvailabilityChangedEvent:
    stt: STT
    available: bool


@dataclass
class _STTStatus:
    available: bool
    recovering_recognize_task: asyncio.Task[None] | None
    recovering_stream_task: asyncio.Task[None] | None


class FallbackAdapter(
    STT[Literal["stt_availability_changed"]],
):
    def __init__(
        self,
        stt: list[STT],
        *,
        vad: VAD | None = None,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        if len(stt) < 1:
            raise ValueError("At least one STT instance must be provided.")

        non_streaming_stt = [t for t in stt if not t.capabilities.streaming]
        if non_streaming_stt:
            if vad is None:
                labels = ", ".join(t.label for t in non_streaming_stt)
                raise ValueError(
                    f"STTs do not support streaming: {labels}. "
                    "Provide a VAD to enable stt.StreamAdapter automatically "
                    "or wrap them with stt.StreamAdapter before using this adapter."
                )
            from ..stt import StreamAdapter

            stt = [
                StreamAdapter(stt=t, vad=vad) if not t.capabilities.streaming else t for t in stt
            ]

        # Use the primary STT's aligned_transcript if all providers support it, since
        # the SDK only checks truthiness, not the specific granularity.
        aligned_transcript: Literal["word", "chunk", False] = False
        if all(t.capabilities.aligned_transcript for t in stt):
            aligned_transcript = stt[0].capabilities.aligned_transcript

        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=all(t.capabilities.interim_results for t in stt),
                diarization=all(t.capabilities.diarization for t in stt),
                aligned_transcript=aligned_transcript,
            )
        )

        self._stt_instances = stt
        self._attempt_timeout = attempt_timeout
        self._current_stt_index: int = 0
        self._max_retry_per_stt = max_retry_per_stt
        self._retry_interval = retry_interval

        self._status: list[_STTStatus] = [
            _STTStatus(
                available=True,
                recovering_recognize_task=None,
                recovering_stream_task=None,
            )
            for _ in self._stt_instances
        ]

        for stt_instance in self._stt_instances:
            stt_instance.on("metrics_collected", self._on_metrics_collected)
        self._recognize_metrics_needed = False  # don't emit metrics via fallback adapter

    @property
    def model(self) -> str:
        return "FallbackAdapter"

    @property
    def provider(self) -> str:
        return "livekit"

    def switch_to_next(self, *, only_if_available: bool = True) -> bool:
        """
        Move pointer to next STT in order.

        Args:
            only_if_available: if True, skip unavailable STTs

        Returns:
            True if switched, False otherwise
        """

        n = len(self._stt_instances)
        start = self._current_stt_index

        for offset in range(1, n + 1):
            i = (start + offset) % n
            status = self._status[i]

            if not only_if_available or status.available:
                prev = self._current_stt_index
                self._current_stt_index = i

                logger.info(
                    f"Manual switch STT from "
                    f"{self._stt_instances[prev].label} "
                    f"to {self._stt_instances[i].label}"
                )

                return True

        return False

    def _ordered_indices(self) -> list[int]:
        n = len(self._stt_instances)
        return (
            [self._current_stt_index]
            + list(range(self._current_stt_index + 1, n))
            + list(range(0, self._current_stt_index))
        )


    def _mark_failed(self, idx: int) -> None:
        self._status[idx].available = False

        for i in self._ordered_indices()[1:]:
            if self._status[i].available:
                self._current_stt_index = i
                logger.info(
                    f"Auto switch STT from "
                    f"{self._stt_instances[idx].label} "
                    f"to {self._stt_instances[i].label}"
                )
                return

    async def _try_recognize(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
        recovering: bool = False,
    ) -> SpeechEvent:
        req_id = str(uuid.uuid4())
        try:
            logger.info(f"[{req_id}] STT Request: Sending recognize request to {stt.label} (language={language})")
            stt_result = await stt.recognize(
                buffer,
                language=language,
                conn_options=dataclasses.replace(
                    conn_options,
                    max_retry=self._max_retry_per_stt,
                    timeout=self._attempt_timeout,
                    retry_interval=self._retry_interval,
                ),
            )
            logger.info(f"[{req_id}] STT Response: Received response from {stt.label}: {stt_result}")
            return stt_result
        except asyncio.TimeoutError:
            if recovering:
                logger.warning(f"[{req_id}] {stt.label} recovery timed out", extra={"streamed": False})
                raise

            logger.warning(
                f"[{req_id}] {stt.label} timed out, switching to next STT",
                extra={"streamed": False},
            )

            raise
        except APIError as e:
            if recovering:
                logger.warning(
                    f"[{req_id}] {stt.label} recovery failed",
                    exc_info=e,
                    extra={"streamed": False},
                )
                raise

            logger.warning(
                f"[{req_id}] {stt.label} failed, switching to next STT",
                exc_info=e,
                extra={"streamed": False},
            )
            raise
        except Exception:
            if recovering:
                logger.exception(
                    f"[{req_id}] {stt.label} recovery unexpected error", extra={"streamed": False}
                )
                raise

            logger.exception(
                f"[{req_id}] {stt.label} unexpected error, switching to next STT",
                extra={"streamed": False},
            )
            raise

    def _try_recovery(
        self,
        *,
        stt: STT,
        buffer: utils.AudioBuffer,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        stt_status = self._status[self._stt_instances.index(stt)]
        if (
            stt_status.recovering_recognize_task is None
            or stt_status.recovering_recognize_task.done()
        ):

            async def _recover_stt_task(stt: STT) -> None:
                try:
                    logger.info(f"Trying to recover {stt.label}...")
                    await self._try_recognize(
                        stt=stt,
                        buffer=buffer,
                        language=language,
                        conn_options=conn_options,
                        recovering=True,
                    )

                    stt_status.available = True
                    logger.info(f"{stt.label} recovered")
                    self.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=True),
                    )
                except Exception:
                    logger.debug(f"{stt.label} recovery attempt failed", exc_info=True)
                    return

            stt_status.recovering_recognize_task = asyncio.create_task(_recover_stt_task(stt))

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        start_time = time.time()

        all_failed = all(not stt_status.available for stt_status in self._status)
        if all_failed:
            logger.error("all STTs are unavailable, retrying..")

        indices = (
            range(len(self._stt_instances))
            if all_failed
            else self._ordered_indices()
        )

        for i in indices:
            stt = self._stt_instances[i]
            stt_status = self._status[i]
            if stt_status.available or all_failed:
                try:
                    return await self._try_recognize(
                        stt=stt,
                        buffer=buffer,
                        language=language,
                        conn_options=conn_options,
                        recovering=False,
                    )
                except Exception:  # exceptions already logged inside _try_recognize
                    if stt_status.available:
                        self._mark_failed(i)
                        self.emit(
                            "stt_availability_changed",
                            AvailabilityChangedEvent(stt=stt, available=False),
                        )

            if not stt_status.available:
                self._try_recovery(
                    stt=stt,
                    buffer=buffer,
                    language=language,
                    conn_options=conn_options,
                )

        raise APIConnectionError(
            f"all STTs failed ({[stt.label for stt in self._stt_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await super().recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return FallbackRecognizeStream(stt=self, language=language, conn_options=conn_options)

    async def aclose(self) -> None:
        for stt_status in self._status:
            if stt_status.recovering_recognize_task is not None:
                await aio.cancel_and_wait(stt_status.recovering_recognize_task)

            if stt_status.recovering_stream_task is not None:
                await aio.cancel_and_wait(stt_status.recovering_stream_task)

        for stt in self._stt_instances:
            stt.off("metrics_collected", self._on_metrics_collected)

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)


class FallbackRecognizeStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: FallbackAdapter,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=NOT_GIVEN)
        self._language = language
        self._fallback_adapter = stt
        self._recovering_streams: list[RecognizeStream] = []

    async def _run(self) -> None:
        start_time = time.time()

        all_failed = all(not stt_status.available for stt_status in self._fallback_adapter._status)
        if all_failed:
            logger.error("all STTs are unavailable, retrying..")

        main_stream: RecognizeStream | None = None
        forward_input_task: asyncio.Task[None] | None = None

        async def _forward_input_task() -> None:
            frame_counter = 0
            async for data in self._input_ch:
                for stream in list(self._recovering_streams):
                    try:
                        if isinstance(data, rtc.AudioFrame):
                            logger.debug(f"STT Stream Request: Pushing audio frame to recovering stream ({stream._stt.label})")
                            stream.push_frame(data)
                        elif isinstance(data, self._FlushSentinel):
                            logger.info(f"STT Stream Request: Flushing recovering stream ({stream._stt.label})")
                            stream.flush()
                    except Exception:
                        pass

                if main_stream is not None:
                    try:
                        if isinstance(data, rtc.AudioFrame):
                            frame_counter += 1
                            # Check if we hit the frame limit (e.g., ~5 seconds)
                            if frame_counter >= 200:
                                logger.debug(f"STT Stream Request: Pushing audio frame to main stream ({main_stream._stt.label})")
                                frame_counter = 0 # Reset the counter
                            main_stream.push_frame(data)
                        elif isinstance(data, self._FlushSentinel):
                            logger.info(f"STT Stream Request: Flushing main stream ({main_stream._stt.label})")
                            main_stream.flush()
                    except Exception:
                        logger.exception(
                            "error happened in forwarding input", extra={"streamed": True}
                        )

            if main_stream is not None:
                with contextlib.suppress(RuntimeError):
                    logger.info(f"STT Stream Request: Ending input for main stream ({main_stream._stt.label})")
                    main_stream.end_input()

        adapter = self._fallback_adapter

        indices = (
            range(len(adapter._stt_instances))
            if all_failed
            else adapter._ordered_indices()
        )

        for i in indices:
            stt = adapter._stt_instances[i]
            stt_status = adapter._status[i]
            if stt_status.available or all_failed:
                req_id = str(uuid.uuid4())
                try:
                    logger.info(f"[{req_id}] STT Stream Request: Starting stream for {stt.label} (language={self._language})")
                    main_stream = stt.stream(
                        language=self._language,
                        conn_options=dataclasses.replace(
                            self._conn_options,
                            max_retry=self._fallback_adapter._max_retry_per_stt,
                            timeout=self._fallback_adapter._attempt_timeout,
                            retry_interval=self._fallback_adapter._retry_interval,
                        ),
                    )

                    if forward_input_task is None or forward_input_task.done():
                        forward_input_task = asyncio.create_task(_forward_input_task())

                    try:
                        async with main_stream:
                            async for ev in main_stream:
                                logger.info(f"[{req_id}] STT Stream Response: Received event from {stt.label}: {ev}")
                                self._event_ch.send_nowait(ev)

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[{req_id}] {stt.label} timed out, switching to next STT",
                            extra={"streamed": True},
                        )
                        raise
                    except APIError as e:
                        logger.warning(
                            f"[{req_id}] {stt.label} failed, switching to next STT",
                            exc_info=e,
                            extra={"streamed": True},
                        )
                        raise
                    except Exception:
                        logger.exception(
                            f"[{req_id}] {stt.label} unexpected error, switching to next STT",
                            extra={"streamed": True},
                        )
                        raise

                    return
                except Exception:
                    if stt_status.available:
                        adapter._mark_failed(i)
                        self._stt.emit(
                            "stt_availability_changed",
                            AvailabilityChangedEvent(stt=stt, available=False),
                        )

            if not stt_status.available:
                self._try_recovery(stt)

        if forward_input_task is not None:
            await aio.cancel_and_wait(forward_input_task)

        await asyncio.gather(*[stream.aclose() for stream in self._recovering_streams])

        raise APIConnectionError(
            f"all STTs failed ({[stt.label for stt in self._fallback_adapter._stt_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
        )

    def _try_recovery(self, stt: STT) -> None:
        stt_status = self._fallback_adapter._status[
            self._fallback_adapter._stt_instances.index(stt)
        ]
        if stt_status.recovering_stream_task is None or stt_status.recovering_stream_task.done():
            req_id = str(uuid.uuid4())
            logger.info(f"[{req_id}] STT Recovery Stream Request: Starting stream for {stt.label}")
            stream = stt.stream(
                language=self._language,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=0,
                    timeout=self._fallback_adapter._attempt_timeout,
                ),
            )
            self._recovering_streams.append(stream)

            async def _recover_stt_task() -> None:
                try:
                    nb_transcript = 0
                    async with stream:
                        async for ev in stream:
                            logger.info(f"[{req_id}] STT Recovery Stream Response: Received event from {stt.label}: {ev}")
                            if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                                if not ev.alternatives or not ev.alternatives[0].text:
                                    continue

                                nb_transcript += 1
                                break

                    if nb_transcript == 0:
                        return

                    stt_status.available = True
                    logger.info(f"stt.FallbackAdapter, {stt.label} recovered")
                    self._fallback_adapter.emit(
                        "stt_availability_changed",
                        AvailabilityChangedEvent(stt=stt, available=True),
                    )

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{req_id}] {stream._stt.label} recovery timed out",
                        extra={"streamed": True},
                    )
                except APIError as e:
                    logger.warning(
                        f"[{req_id}] {stream._stt.label} recovery failed",
                        exc_info=e,
                        extra={"streamed": True},
                    )
                except Exception:
                    logger.exception(
                        f"[{req_id}] {stream._stt.label} recovery unexpected error",
                        extra={"streamed": True},
                    )
                    raise

            stt_status.recovering_stream_task = task = asyncio.create_task(_recover_stt_task())
            task.add_done_callback(lambda _: self._recovering_streams.remove(stream))

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        async for _ in event_aiter:
            pass
