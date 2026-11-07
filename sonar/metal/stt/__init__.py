# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speech-to-Text support for Sonar Metal."""

from sonar.metal.stt.loader import load_model
from sonar.metal.stt.protocol import TranscriptionResult, TranscriptionSegment
from sonar.metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber
from sonar.metal.stt.whisper import WhisperTranscriber

__all__ = [
    "Qwen3ASRTranscriber",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperTranscriber",
    "load_model",
]
