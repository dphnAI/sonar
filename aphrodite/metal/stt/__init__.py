# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for Aphrodite Metal."""

from aphrodite.metal.stt.loader import load_model
from aphrodite.metal.stt.protocol import TranscriptionResult, TranscriptionSegment
from aphrodite.metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber
from aphrodite.metal.stt.whisper import WhisperTranscriber

__all__ = [
    "Qwen3ASRTranscriber",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperTranscriber",
    "load_model",
]
