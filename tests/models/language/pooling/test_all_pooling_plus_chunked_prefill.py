# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModel

from tests.models.utils import check_embeddings_close
from aphrodite import TokensPrompt
from aphrodite.config import PoolerConfig


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-Embedding-0.6B"],
)
@torch.inference_mode
def test_embed_models(hf_runner, aphrodite_runner, model: str):
    chunk_size = 10
    n_prompt_tokens = [55, 56, 57]
    token_prompts = [[1024 + i for i in range(n)] for n in n_prompt_tokens]

    with aphrodite_runner(
        model,
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        max_model_len=128,
        max_num_batched_tokens=chunk_size,
        enforce_eager=True,
        # `enable_chunked_prefill`: Set to `False` instead of `None` in AphroditeRunner
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.token_embed(
            [TokensPrompt(prompt_token_ids=t) for t in token_prompts],
        )

    with hf_runner(
        model,
        auto_cls=AutoModel,
    ) as hf_model:
        hf_outputs = []
        for token_prompt in token_prompts:
            inputs = hf_model.wrap_device({"input_ids": torch.tensor([token_prompt])})
            input_ids = inputs["input_ids"]
            output = hf_model.model(input_ids)
            hf_outputs.append(output.last_hidden_state.cpu().float()[0])

    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        check_embeddings_close(
            embeddings_0_lst=hf_output,
            embeddings_1_lst=aphrodite_output,
            name_0="hf",
            name_1="aphrodite",
            tol=1e-2,
        )
