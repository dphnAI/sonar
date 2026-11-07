# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
from transformers import AutoTokenizer

from sonar.config import DeviceConfig, SonarConfig
from sonar.v1.structured_output.backend_guidance import GuidanceBackend
from sonar.v1.structured_output.backend_lm_format_enforcer import (
    LMFormatEnforcerBackend,
)
from sonar.v1.structured_output.backend_outlines import OutlinesBackend
from sonar.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputOptions,
)
from sonar.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "gpt2"
REGEX = "1a"
BACKENDS = (
    pytest.param("xgrammar", id="xgrammar"),
    pytest.param("guidance", id="guidance"),
    pytest.param("outlines", id="outlines"),
    pytest.param("lm-format-enforcer", id="lm-format-enforcer"),
)
BACKEND_CLASSES: dict[str, Callable[..., StructuredOutputBackend]] = {
    "xgrammar": XgrammarBackend,
    "guidance": GuidanceBackend,
    "outlines": OutlinesBackend,
    "lm-format-enforcer": LMFormatEnforcerBackend,
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _single_token(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text)
    assert len(token_ids) == 1, (text, token_ids)
    return token_ids[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_accept_tokens_rejection_is_atomic(tokenizer, backend: str):
    structured_output_backend = BACKEND_CLASSES[backend](
        SonarConfig(device_config=DeviceConfig(device="cpu")),
        tokenizer,
        len(tokenizer),
    )
    grammar = structured_output_backend.compile_grammar(StructuredOutputOptions.REGEX, REGEX)

    first = _single_token(tokenizer, "1")
    invalid = _single_token(tokenizer, "z")

    assert not grammar.accept_tokens("request", [first, invalid])
    assert grammar.accept_tokens("request", [first])


@pytest.mark.parametrize("wrapper", ["group", "wrapper", "nested"])
def test_lmfe_wrapped_tokenizer_enforces_grammar(tokenizer, wrapper):
    class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    class TokenizerGroup:
        def __init__(self, tokenizer):
            self._tokenizer = tokenizer

        def get_tokenizer(self):
            return self._tokenizer

    supplied = TokenizerWrapper(tokenizer) if wrapper in ("wrapper", "nested") else tokenizer
    if wrapper in ("group", "nested"):
        supplied = TokenizerGroup(supplied)

    backend = LMFormatEnforcerBackend(SonarConfig(device_config=DeviceConfig(device="cpu")), supplied, len(tokenizer))
    grammar = backend.compile_grammar(StructuredOutputOptions.REGEX, REGEX)
    first = _single_token(tokenizer, "1")
    invalid = _single_token(tokenizer, "z")
    last = _single_token(tokenizer, "a")
    assert not grammar.accept_tokens("request", [first, invalid])
    assert grammar.accept_tokens("request", [first, last])


def test_lmfe_padded_model_vocab_keeps_invalid_logits_masked(tokenizer):
    vocab_size = len(tokenizer) + 64
    backend = LMFormatEnforcerBackend(SonarConfig(device_config=DeviceConfig(device="cpu")), tokenizer, vocab_size)
    grammar = backend.compile_grammar(StructuredOutputOptions.CHOICE, '["0", "1"]')
    bitmask = backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)

    def is_allowed(token_id):
        return bool(int(bitmask[0, token_id // 32]) & (1 << (token_id % 32)))

    assert is_allowed(_single_token(tokenizer, "0"))
    assert is_allowed(_single_token(tokenizer, "1"))
    assert not is_allowed(_single_token(tokenizer, "z"))
    assert not any(is_allowed(token_id) for token_id in range(len(tokenizer), vocab_size))
    assert grammar.accept_tokens("request", tokenizer.encode("1"))
    grammar.fill_bitmask(bitmask, 0)
    assert is_allowed(tokenizer.eos_token_id)
