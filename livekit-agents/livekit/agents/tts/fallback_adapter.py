from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from livekit import rtc

from .. import utils
from .._exceptions import APIConnectionError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, USERDATA_TIMED_TRANSCRIPT, APIConnectOptions
from ..utils import aio
from .stream_adapter import StreamAdapter
from .tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)

# don't retry when using the fallback adapter
DEFAULT_FALLBACK_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class _TTSStatus:
    available: bool
    recovering_task: asyncio.Task[None] | None
    needs_resampling: bool


@dataclass
class AvailabilityChangedEvent:
    tts: TTS
    available: bool


class FallbackAdapter(
    TTS[Literal["tts_availability_changed"]],
):
    """
    Manages multiple TTS instances, providing a fallback mechanism to ensure continuous TTS service.
    """

    def __init__(
        self,
        tts: list[TTS],
        *,
        max_retry_per_tts: int = 2,
        sample_rate: int | None = None,
    ) -> None:
        """
        Initialize a FallbackAdapter that manages multiple TTS instances.

        Args:
            tts (list[TTS]): A list of TTS instances to use for fallback.
            max_retry_per_tts (int, optional): Maximum number of retries per TTS instance. Defaults to 2.
            sample_rate (int | None, optional): Desired sample rate for the synthesized audio. If None, uses the maximum sample rate among the TTS instances.

        Raises:
            ValueError: If less than one TTS instance is provided.
            ValueError: If TTS instances have different numbers of channels.
        """  # noqa: E501

        if len(tts) < 1:
            raise ValueError("at least one TTS instance must be provided.")

        if len({t.num_channels for t in tts}) != 1:
            raise ValueError("all TTS must have the same number of channels")

        if sample_rate is None:
            sample_rate = max(t.sample_rate for t in tts)

        num_channels = tts[0].num_channels

        super().__init__(
            capabilities=TTSCapabilities(
                streaming=any(t.capabilities.streaming for t in tts),
                aligned_transcript=all(t.capabilities.aligned_transcript for t in tts),
            ),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._tts_instances = tts
        self._max_retry_per_tts = max_retry_per_tts
        self._current_tts_index: int = 0

        self._status: list[_TTSStatus] = []
        for t in tts:
            needs_resampling = sample_rate != t.sample_rate
            if needs_resampling:
                logger.info(f"resampling {t.label} from {t.sample_rate}Hz to {sample_rate}Hz")

            self._status.append(
                _TTSStatus(available=True, recovering_task=None, needs_resampling=needs_resampling)
            )

            t.on("metrics_collected", self._on_metrics_collected)

    @property
    def model(self) -> str:
        return "FallbackAdapter"

    @property
    def provider(self) -> str:
        return "livekit"

    def _ordered_indices(self) -> list[int]:
        n = len(self._tts_instances)
        return (
            [self._current_tts_index]
            + list(range(self._current_tts_index + 1, n))
            + list(range(0, self._current_tts_index))
        )

    def _mark_failed(self, idx: int) -> None:
        self._status[idx].available = False

        for i in self._ordered_indices()[1:]:
            if self._status[i].available:
                self._current_tts_index = i
                logger.info(
                    f"Auto switch TTS from "
                    f"{self._tts_instances[idx].label} "
                    f"to {self._tts_instances[i].label}"
                )
                return

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS
    ) -> FallbackChunkedStream:
        return FallbackChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS
    ) -> FallbackSynthesizeStream:
        return FallbackSynthesizeStream(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        if self._tts_instances:
            self._tts_instances[0].prewarm()

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def aclose(self) -> None:
        for tts_status in self._status:
            if tts_status.recovering_task is not None:
                await aio.cancel_and_wait(tts_status.recovering_task)

        for t in self._tts_instances:
            t.off("metrics_collected", self._on_metrics_collected)


class FallbackChunkedStream(ChunkedStream):
    _tts_request_span_name: ClassVar[str] = "tts_fallback_adapter"

    def __init__(
        self, *, tts: FallbackAdapter, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._fallback_adapter = tts

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        pass  # do nothing

    async def _try_synthesize(
        self, *, tts: TTS, recovering: bool = False
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        try:
            async with tts.synthesize(
                self._input_text,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=self._fallback_adapter._max_retry_per_tts,
                    timeout=self._conn_options.timeout,
                    retry_interval=self._conn_options.retry_interval,
                ),
            ) as stream:
                async for audio in stream:
                    yield audio

        except Exception as e:
            if recovering:
                logger.warning(
                    f"{tts.label} recovery failed", extra={"streamed": False}, exc_info=e
                )
                raise

            logger.warning(
                f"{tts.label} error, switching to next TTS",
                extra={"streamed": False},
            )
            raise

    def _try_recovery(self, tts: TTS) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        tts_status = self._tts._status[self._tts._tts_instances.index(tts)]
        if tts_status.recovering_task is None or tts_status.recovering_task.done():

            async def _recover_tts_task(tts: TTS) -> None:
                try:
                    async for _ in self._try_synthesize(tts=tts, recovering=True):
                        pass

                    tts_status.available = True
                    logger.info(f"tts.FallbackAdapter, {tts.label} recovered")
                    self._tts.emit(
                        "tts_availability_changed",
                        AvailabilityChangedEvent(tts=tts, available=True),
                    )
                except Exception:  # exceptions already logged inside _try_synthesize
                    return

            tts_status.recovering_task = asyncio.create_task(_recover_tts_task(tts))

    async def _run(self, output_emitter: AudioEmitter) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        start_time = time.time()

        all_failed = all(not tts_status.available for tts_status in self._tts._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )

        adapter = self._tts

        indices = (
            range(len(adapter._tts_instances))
            if all_failed
            else adapter._ordered_indices()
        )

        for i in indices:
            tts = adapter._tts_instances[i]
            tts_status = self._tts._status[i]
            if tts_status.available or all_failed:
                logger.debug(f"tts.FallbackAdapter request_id={request_id} sending request to {tts.label}")
                try:
                    resampler = (
                        rtc.AudioResampler(
                            input_rate=tts.sample_rate,
                            output_rate=self._tts.sample_rate,
                        )
                        if tts_status.needs_resampling
                        else None
                    )
                    async for synthesized_audio in self._try_synthesize(tts=tts, recovering=False):
                        if texts := synthesized_audio.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                            output_emitter.push_timed_transcript(texts)

                        if resampler is not None:
                            for rf in resampler.push(synthesized_audio.frame):
                                output_emitter.push(rf.data.tobytes())
                        else:
                            output_emitter.push(synthesized_audio.frame.data.tobytes())

                    if resampler is not None:
                        for rf in resampler.flush():
                            output_emitter.push(rf.data.tobytes())

                    logger.debug(f"tts.FallbackAdapter request_id={request_id} received response from {tts.label} successfully after {time.time() - start_time} seconds")
                    return
                except Exception:  # exceptions already logged inside _try_synthesize
                    logger.warning(f"tts.FallbackAdapter request_id={request_id} request to {tts.label} failed")
                    if tts_status.available:
                        adapter._mark_failed(i)
                        self._tts.emit(
                            "tts_availability_changed",
                            AvailabilityChangedEvent(tts=tts, available=False),
                        )

                    if output_emitter.pushed_duration() > 0.0:
                        logger.warning(
                            f"{tts.label} already synthesized of audio, ignoring fallback"
                        )
                        return

            if not tts_status.available:
                self._try_recovery(tts)

        raise APIConnectionError(
            f"all TTSs failed ({[tts.label for tts in self._tts._tts_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
        )


class FallbackSynthesizeStream(SynthesizeStream):
    _tts_request_span_name: ClassVar[str] = "tts_fallback_adapter"

    def __init__(self, *, tts: FallbackAdapter, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._fallback_adapter = tts
        self._pushed_tokens: list[str] = []

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        pass  # do nothing

    async def _try_synthesize(
        self,
        *,
        tts: TTS,
        input_ch: aio.ChanReceiver[str | SynthesizeStream._FlushSentinel],
        conn_options: APIConnectOptions,
        recovering: bool = False,
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        # If TTS doesn't support streaming, wrap it with StreamAdapter
        if tts.capabilities.streaming:
            stream = tts.stream(conn_options=conn_options)
        else:
            from .. import tokenize

            wrapped_tts = StreamAdapter(
                tts=tts,
                sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(retain_format=True),
            )
            stream = wrapped_tts.stream(conn_options=conn_options)

        @utils.log_exceptions(logger=logger)
        async def _forward_input_task() -> None:
            try:
                async for data in input_ch:
                    if isinstance(data, str):
                        stream.push_text(data)
                    elif isinstance(data, self._FlushSentinel):
                        stream.flush()
            finally:
                stream.end_input()

        input_task = asyncio.create_task(_forward_input_task())

        try:
            async with stream:
                async for audio in stream:
                    yield audio
        except Exception as e:
            if recovering:
                logger.warning(
                    f"{tts.label} recovery failed",
                    extra={"streamed": True},
                    exc_info=e,
                )
                raise

            logger.exception(
                f"{tts.label} error, switching to next TTS",
                extra={"streamed": True},
            )
            raise
        finally:
            await utils.aio.cancel_and_wait(input_task)

    async def _run(self, output_emitter: AudioEmitter) -> None:
        start_time = time.time()

        all_failed = all(not tts_status.available for tts_status in self._fallback_adapter._status)
        if all_failed:
            logger.error("all TTSs are unavailable, retrying..")

        new_input_ch: aio.Chan[str | SynthesizeStream._FlushSentinel] | None = None
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._fallback_adapter.sample_rate,
            num_channels=self._fallback_adapter.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())

        async def _forward_input_task() -> None:
            nonlocal new_input_ch

            async for data in self._input_ch:
                if new_input_ch:
                    new_input_ch.send_nowait(data)

                if isinstance(data, str) and data:
                    self._pushed_tokens.append(data)

            if new_input_ch:
                new_input_ch.close()

        input_task = asyncio.create_task(_forward_input_task())

        try:
            adapter = self._fallback_adapter

            indices = (
                range(len(adapter._tts_instances))
                if all_failed
                else adapter._ordered_indices()
            )

            for i in indices:
                tts = adapter._tts_instances[i]
                tts_status = self._fallback_adapter._status[i]
                if tts_status.available or all_failed:
                    logger.debug(f"tts.FallbackAdapter request_id={request_id} sending streaming request to {tts.label}")
                    try:
                        new_input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()

                        for text in self._pushed_tokens:
                            new_input_ch.send_nowait(text)

                        if input_task.done():
                            new_input_ch.close()

                        resampler = (
                            rtc.AudioResampler(
                                input_rate=tts.sample_rate,
                                output_rate=self._fallback_adapter.sample_rate,
                            )
                            if tts_status.needs_resampling
                            else None
                        )
                        async for synthesized_audio in self._try_synthesize(
                            tts=tts,
                            input_ch=new_input_ch,
                            conn_options=dataclasses.replace(
                                self._conn_options,
                                max_retry=self._fallback_adapter._max_retry_per_tts,
                                timeout=self._conn_options.timeout,
                                retry_interval=self._conn_options.retry_interval,
                            ),
                            recovering=False,
                        ):
                            if texts := synthesized_audio.frame.userdata.get(
                                USERDATA_TIMED_TRANSCRIPT
                            ):
                                output_emitter.push_timed_transcript(texts)

                            if resampler is not None:
                                for resampled_frame in resampler.push(synthesized_audio.frame):
                                    output_emitter.push(resampled_frame.data.tobytes())

                                if synthesized_audio.is_final:
                                    for resampled_frame in resampler.flush():
                                        output_emitter.push(resampled_frame.data.tobytes())
                            else:
                                output_emitter.push(synthesized_audio.frame.data.tobytes())

                        logger.debug(f"tts.FallbackAdapter request_id={request_id} received streaming response from {tts.label} successfully after {time.time() - start_time} seconds")
                        return
                    except Exception:
                        logger.warning(f"tts.FallbackAdapter request_id={request_id} streaming request to {tts.label} failed")
                        if tts_status.available:
                            adapter._mark_failed(i)
                            self._tts.emit(
                                "tts_availability_changed",
                                AvailabilityChangedEvent(tts=tts, available=False),
                            )

                        if output_emitter.pushed_duration() > 0.0:
                            logger.warning(
                                f"{tts.label} already synthesized of audio, ignoring the current segment for the tts fallback"  # noqa: E501
                            )
                            return

                if not tts_status.available:
                    self._try_recovery(tts)

            raise APIConnectionError(
                f"all TTSs failed ({[tts.label for tts in self._fallback_adapter._tts_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
            )
        finally:
            await utils.aio.cancel_and_wait(input_task)

    def _try_recovery(self, tts: TTS) -> None:
        assert isinstance(self._tts, FallbackAdapter)

        retry_text = self._pushed_tokens.copy()
        if not retry_text:
            return

        tts_status = self._tts._status[self._tts._tts_instances.index(tts)]
        if tts_status.recovering_task is None or tts_status.recovering_task.done():

            async def _recover_tts_task(tts: TTS) -> None:
                try:
                    input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
                    for t in retry_text:
                        input_ch.send_nowait(t)

                    input_ch.close()

                    async for _ in self._try_synthesize(
                        tts=tts,
                        input_ch=input_ch,
                        recovering=True,
                        conn_options=dataclasses.replace(
                            self._conn_options,
                            max_retry=0,
                            timeout=self._conn_options.timeout,
                            retry_interval=self._conn_options.retry_interval,
                        ),
                    ):
                        pass

                    tts_status.available = True
                    logger.info(f"tts.FallbackAdapter, {tts.label} recovered")
                    self._tts.emit(
                        "tts_availability_changed",
                        AvailabilityChangedEvent(tts=tts, available=True),
                    )
                except Exception:
                    return

            tts_status.recovering_task = asyncio.create_task(_recover_tts_task(tts))
