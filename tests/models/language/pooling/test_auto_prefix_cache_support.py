# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from tests.models.language.pooling.embed_utils import run_embedding_correctness_test


@pytest.mark.parametrize(
    "model",
    ["vllm-project/Qwen2.5-1.5B-apeach"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    # example_prompts is too short for testing prefix_caching
    example_prompts = [s * 10 for s in example_prompts]

    with sonar_runner(model, max_model_len=512, dtype=dtype, enable_prefix_caching=True) as sonar_model:
        sonar_config = sonar_model.llm.llm_engine.sonar_config
        cache_config = sonar_config.cache_config
        assert cache_config.enable_prefix_caching

        # First Run
        sonar_model.classify(example_prompts)

        # assert prefix_caching works
        pooling_outputs = sonar_model.llm.encode(example_prompts, pooling_task="classify")
        for output in pooling_outputs:
            assert output.num_cached_tokens > 0
        sonar_outputs = [req_output.outputs.data for req_output in pooling_outputs]

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
        hf_output = torch.tensor(hf_output)
        sonar_output = torch.tensor(sonar_output)

        assert torch.allclose(hf_output, sonar_output, 1e-3 if dtype == "float" else 1e-2)


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-Embedding-0.6B"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_embed_models(
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    dtype: str,
):
    # example_prompts is too short for testing prefix_caching
    example_prompts = [str(s).strip() * 10 for s in example_prompts]

    with sonar_runner(
        model,
        runner="pooling",
        max_model_len=None,
        enable_prefix_caching=True,
    ) as sonar_model:
        sonar_config = sonar_model.llm.llm_engine.sonar_config
        cache_config = sonar_config.cache_config
        assert cache_config.enable_prefix_caching

        # First Run
        sonar_model.embed(example_prompts)

        # assert prefix_caching works
        pooling_outputs = sonar_model.llm.encode(example_prompts, pooling_task="embed")
        for output in pooling_outputs:
            assert output.num_cached_tokens > 0
        sonar_outputs = [req_output.outputs.data for req_output in pooling_outputs]

    with hf_runner(
        model,
        is_sentence_transformer=True,
    ) as hf_model:
        run_embedding_correctness_test(hf_model, example_prompts, sonar_outputs)


@pytest.mark.parametrize(
    "model",
    [
        "vllm-project/e5-small",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # is_causal == False
        "vllm-project/xlm-roberta-base-language-detection",
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_non_causal_models(hf_runner, sonar_runner, example_prompts, model: str, dtype: str) -> None:
    with sonar_runner(model, max_model_len=512, dtype=dtype) as sonar_model:
        sonar_config = sonar_model.llm.llm_engine.sonar_config
        cache_config = sonar_config.cache_config
        assert not cache_config.enable_prefix_caching
