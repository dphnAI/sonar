# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModel

from aphrodite.config import PoolerConfig
from tests.models.utils import check_embeddings_close


@pytest.mark.parametrize(
    "model",
    ["BAAI/bge-m3"],
)
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_embed_models(hf_runner, aphrodite_runner, example_prompts, model: str, dtype: str):
    with aphrodite_runner(
        model,
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        max_model_len=None,
        dtype=dtype,
    ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.token_embed(example_prompts)

    with hf_runner(model, auto_cls=AutoModel, dtype=dtype) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            embedding = output.last_hidden_state[0].float()
            hf_outputs.append(embedding.cpu())

    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        check_embeddings_close(
            embeddings_0_lst=hf_output,
            embeddings_1_lst=aphrodite_output,
            name_0="hf",
            name_1="aphrodite",
            tol=1e-2,
        )
