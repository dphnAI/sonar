# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from tests.models.language.pooling.embed_utils import run_embedding_correctness_test


@pytest.mark.parametrize(
    "model",
    ["jason9693/Qwen2.5-1.5B-apeach"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    # example_prompts is too short for testing prefix_caching
    example_prompts = [s * 10 for s in example_prompts]

    with aphrodite_runner(model, max_model_len=512, dtype=dtype, enable_prefix_caching=True) as aphrodite_model:
        aphrodite_config = aphrodite_model.llm.llm_engine.aphrodite_config
        cache_config = aphrodite_config.cache_config
        assert cache_config.enable_prefix_caching

        # First Run
        aphrodite_model.classify(example_prompts)

        # assert prefix_caching works
        pooling_outputs = aphrodite_model.llm.encode(example_prompts, pooling_task="classify")
        for output in pooling_outputs:
            assert output.num_cached_tokens > 0
        aphrodite_outputs = [req_output.outputs.data for req_output in pooling_outputs]

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        hf_output = torch.tensor(hf_output)
        aphrodite_output = torch.tensor(aphrodite_output)

        assert torch.allclose(hf_output, aphrodite_output, 1e-3 if dtype == "float" else 1e-2)


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-Embedding-0.6B"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_embed_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
):
    # example_prompts is too short for testing prefix_caching
    example_prompts = [str(s).strip() * 10 for s in example_prompts]

    with aphrodite_runner(
        model,
        runner="pooling",
        max_model_len=None,
        enable_prefix_caching=True,
    ) as aphrodite_model:
        aphrodite_config = aphrodite_model.llm.llm_engine.aphrodite_config
        cache_config = aphrodite_config.cache_config
        assert cache_config.enable_prefix_caching

        # First Run
        aphrodite_model.embed(example_prompts)

        # assert prefix_caching works
        pooling_outputs = aphrodite_model.llm.encode(example_prompts, pooling_task="embed")
        for output in pooling_outputs:
            assert output.num_cached_tokens > 0
        aphrodite_outputs = [req_output.outputs.data for req_output in pooling_outputs]

    with hf_runner(
        model,
        is_sentence_transformer=True,
    ) as hf_model:
        run_embedding_correctness_test(hf_model, example_prompts, aphrodite_outputs)


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/e5-small",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # is_causal == False
        "papluca/xlm-roberta-base-language-detection",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_non_causal_models(hf_runner, aphrodite_runner, example_prompts, model: str, dtype: str) -> None:
    with aphrodite_runner(model, max_model_len=512, dtype=dtype) as aphrodite_model:
        aphrodite_config = aphrodite_model.llm.llm_engine.aphrodite_config
        cache_config = aphrodite_config.cache_config
        assert not cache_config.enable_prefix_caching
