# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Every OpenAI text endpoint must forward Aphrodite's extra sampler fields.

Regression test: an upstream sync rewrote ChatCompletionRequest and silently
dropped these fields, so chat requests ran with plain sampling while returning
HTTP 200.
"""

from types import SimpleNamespace

import pytest

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from aphrodite.entrypoints.openai.completion.protocol import CompletionRequest
from aphrodite.entrypoints.openai.kobold.protocol import KAIGenerationInputSchema
from aphrodite.entrypoints.openai.kobold.serving import OpenAIServingKobold

pytestmark = pytest.mark.cpu_test

# Non-default value for every extra field.
EXTENDED = {
    "top_a": 0.2,
    "tfs": 0.9,
    "eta_cutoff": 3.0,
    "epsilon_cutoff": 0.01,
    "typical_p": 0.8,
    "smoothing_factor": 0.3,
    "smoothing_curve": 1.5,
    "no_repeat_ngram_size": 3,
    "temperature_last": True,
    "xtc_threshold": 0.2,
    "xtc_probability": 0.5,
    "dry_multiplier": 0.8,
    "dry_base": 2.0,
    "dry_allowed_length": 3,
    "dry_sequence_breakers": ["\n", ":"],
    "dry_range": 64,
    "dry_max_ngram": 10,
    "dry_max_occurrences": 6,
    "dry_early_exit_match_len": 5,
    "dynatemp_min": 0.5,
    "dynatemp_max": 1.5,
    "dynatemp_exponent": 2.0,
    "nsigma": 1.0,
    "skew": 0.5,
    "mirostat_mode": 2,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
}
MESSAGES = [{"role": "user", "content": "hi"}]


def _completion(**fields):
    return CompletionRequest(model="m", prompt="hi", **fields).to_sampling_params(16, {})


def _chat(**fields):
    return ChatCompletionRequest(model="m", messages=MESSAGES, **fields).to_sampling_params(16, {})


def _batch_chat(**fields):
    request = BatchChatCompletionRequest(model="m", messages=[MESSAGES], **fields)
    return request.to_chat_completion_request(MESSAGES).to_sampling_params(16, {})


ENDPOINTS = [
    pytest.param(_completion, id="completion"),
    pytest.param(_chat, id="chat"),
    pytest.param(_batch_chat, id="batch_chat"),
]


@pytest.mark.parametrize("build", ENDPOINTS)
def test_extended_sampler_fields_reach_sampling_params(build):
    params = build(**EXTENDED)
    assert {name: getattr(params, name) for name in EXTENDED} == EXTENDED


@pytest.mark.parametrize("build", ENDPOINTS)
def test_endpoints_share_extended_sampler_defaults(build):
    reference = _completion()
    params = build()
    assert {name: getattr(params, name) for name in EXTENDED} == {name: getattr(reference, name) for name in EXTENDED}


@pytest.mark.parametrize("build", ENDPOINTS)
def test_stringified_dry_sequence_breakers_are_parsed(build):
    assert build(dry_sequence_breakers='["\\n", ":"]').dry_sequence_breakers == ["\n", ":"]


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def _kobold(**fields):
    payload = KAIGenerationInputSchema(prompt="hi", max_context_length=64, max_length=8, **fields)
    params, _ = OpenAIServingKobold._prepare_engine_payload(SimpleNamespace(badwordsids=[]), payload, _Tokenizer())
    return params


def test_kobold_temperature_zero_is_greedy():
    params = _kobold(temperature=0.0, top_k=40, top_p=0.9, dynatemp_range=0.5)
    assert params.temperature == 0.0
    assert (params.top_k, params.top_p) == (0, 1.0)
    assert (params.dynatemp_min, params.dynatemp_max) == (0.0, 0.0)


def test_kobold_dynatemp_range_spans_temperature():
    params = _kobold(temperature=0.7, dynatemp_range=0.5, dynatemp_exponent=1.5)
    assert params.dynatemp_min == pytest.approx(0.2)
    assert params.dynatemp_max == pytest.approx(1.2)
    assert params.dynatemp_exponent == 1.5
    clamped = _kobold(temperature=0.3, dynatemp_range=0.5)
    assert clamped.dynatemp_min == 0.0
    assert clamped.dynatemp_max == pytest.approx(0.8)
