# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from aphrodite.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from aphrodite.model_executor.models.moss_transcribe_diarize import (
    DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT,
    MossTranscribeDiarizeForConditionalGeneration,
)


def _parse(text: str):
    return MossTranscribeDiarizeForConditionalGeneration.parse_diarized_transcript(text)


def _question(request_prompt: str = "", hotwords: str | None = None) -> str:
    params = SpeechToTextParams(
        audio=np.zeros(1600, dtype=np.float32),
        stt_config=SpeechToTextConfig(),
        model_config=None,
        hotwords=hotwords,
        request_prompt=request_prompt,
    )
    prompt_data = MossTranscribeDiarizeForConditionalGeneration.get_generation_prompt(params)
    assert isinstance(prompt_data, dict)
    prompt = prompt_data.get("prompt")
    assert isinstance(prompt, str)
    return prompt.split("<|audio_end|>\n", 1)[1].split("<|im_end|>", 1)[0]


@pytest.mark.parametrize(
    ("request_prompt", "hotwords", "expected"),
    [
        ("", None, DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT),
        ("", "  ", DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT),
        (
            "",
            "魔搭, 达摩院",
            DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT + "热词提示：魔搭, 达摩院",
        ),
        ("转录为文本。", " Aphrodite ", "转录为文本。热词提示：Aphrodite"),
    ],
    ids=[
        "default",
        "blank-hotwords",
        "hotwords",
        "custom-prompt-hotwords",
    ],
)
def test_generation_prompt_appends_hotwords(request_prompt, hotwords, expected):
    assert _question(request_prompt, hotwords) == expected


def test_parse_diarized_transcript_preserves_moss_segments():
    segments = _parse("[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]")
    assert [(s.start, s.end, s.speaker, s.text) for s in segments] == [
        (0.48, 1.66, "S01", "Welcome"),
        (12.26, 13.81, "S02", "Ready"),
    ]


def test_parse_diarized_transcript_preserves_overlapping_segments():
    segments = _parse("[0][S01]First speaker[2][1][S02]Second speaker[3]")
    assert [(s.start, s.end, s.speaker, s.text) for s in segments] == [
        (0.0, 2.0, "S01", "First speaker"),
        (1.0, 3.0, "S02", "Second speaker"),
    ]


def test_parse_diarized_transcript_preserves_numeric_text_markers():
    assert [s.text for s in _parse("[0][S01]The [2024] report is ready.[4]")] == ["The [2024] report is ready."]


def test_parse_diarized_transcript_ignores_whitespace_between_segments():
    segments = _parse("[0][S01]Hello[1]\n [2][S02]Hi[3]")
    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 1.0, "Hello"),
        (2.0, 3.0, "Hi"),
    ]


def test_parse_diarized_transcript_ignores_noise_before_a_segment():
    segments = _parse("noise [bad][0.1][S01]Hello[0.9]")
    assert [(s.start, s.end, s.text) for s in segments] == [(0.1, 0.9, "Hello")]


def test_parse_diarized_transcript_preserves_timestamps_before_the_end():
    assert [s.text for s in _parse("[2][S01]The earlier timestamp is [1] not the end[3]")] == [
        "The earlier timestamp is [1] not the end"
    ]


def test_parse_diarized_transcript_skips_empty_segments():
    segments = _parse("[0][S01][1][2][S02]Complete[3]")
    assert [(s.speaker, s.text) for s in segments] == [("S02", "Complete")]


def test_parse_diarized_transcript_fails_closed_for_incomplete_output():
    assert _parse("[0][S01]Complete[1][2][S02]Incomplete") == []


def test_parse_diarized_transcript_fails_closed_for_trailing_text():
    assert _parse("[0][S01]Complete[1] trailing text") == []


def test_parse_diarized_transcript_preserves_overlong_timestamp_markers():
    marker = f"[{'1' * 33}]"
    assert [s.text for s in _parse(f"[0][S01]Value {marker}[1]")] == [f"Value {marker}"]
