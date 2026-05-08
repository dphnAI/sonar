# SPDX-License-Identifier: Apache-2.0
"""Whisper STT implementation (model-owned package)."""

from .config import WhisperConfig
from .model import WhisperModel
from .transcriber import WhisperTranscriber

__all__ = [
    "WhisperConfig",
    "WhisperModel",
    "WhisperTranscriber",
]
