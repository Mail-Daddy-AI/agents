# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Silero VAD plugin for LiveKit Agents

See https://docs.livekit.io/agents/build/turns/vad/ for more information.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .vad import VAD, VADStream
from .version import __version__

__all__ = ["VAD", "VADStream", "__version__"]

from livekit.agents import Plugin

from .log import logger

# Same model LiveKit ships via Git LFS in the source tree; mirrors snakers4/silero-vad.
_SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_MIN_ONNX_BYTES = 100_000


def _silero_onnx_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "silero_vad.onnx"


def _onnx_looks_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size < _MIN_ONNX_BYTES:
        return False
    with path.open("rb") as f:
        head = f.read(32)
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False
    return True


class SileroPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        path = _silero_onnx_path()
        if _onnx_looks_valid(path):
            logger.info("silero_vad.onnx already present, skipping download")
            return
        logger.info("Downloading silero_vad.onnx for livekit.plugins.silero")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(_SILERO_VAD_URL, timeout=120) as resp:  # noqa: S310
                tmp.write_bytes(resp.read())
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        logger.info("Finished downloading silero_vad.onnx")


Plugin.register_plugin(SileroPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
