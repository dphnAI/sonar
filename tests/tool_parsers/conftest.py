# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from aphrodite.tokenizers import TokenizerLike


@pytest.fixture(scope="module")
def default_tokenizer() -> TokenizerLike:
    return AutoTokenizer.from_pretrained("gpt2")
