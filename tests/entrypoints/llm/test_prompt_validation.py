# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from aphrodite.exceptions import AphroditeValidationError


def test_empty_prompt(aphrodite_runner):
    with (
        aphrodite_runner("openai-community/gpt2", enforce_eager=True) as runner,
        pytest.raises(AphroditeValidationError, match="decoder prompt cannot be empty"),
    ):
        runner.llm.generate([""])


def test_out_of_vocab_token(aphrodite_runner):
    with (
        aphrodite_runner("openai-community/gpt2", enforce_eager=True) as runner,
        pytest.raises(AphroditeValidationError, match="out of vocabulary"),
    ):
        runner.llm.generate({"prompt_token_ids": [999999]})


def test_out_of_vocab_token_skip_tokenizer_init(aphrodite_runner):
    # Without a tokenizer the model vocab size is the only bound; an
    # out-of-range id must not reach the embedding lookup.
    with (
        aphrodite_runner(
            "openai-community/gpt2",
            enforce_eager=True,
            skip_tokenizer_init=True,
        ) as runner,
        pytest.raises(AphroditeValidationError, match="out of vocabulary"),
    ):
        runner.llm.generate({"prompt_token_ids": [999999]})


def test_require_mm_embeds(aphrodite_runner):
    with (
        aphrodite_runner(
            "llava-hf/llava-1.5-7b-hf",
            enforce_eager=True,
            enable_mm_embeds=False,
        ) as runner,
        pytest.raises(ValueError, match="--enable-mm-embeds"),
    ):
        runner.llm.generate(
            {
                "prompt": "<image>",
                "multi_modal_data": {"image": torch.empty(1, 1, 1)},
            }
        )
