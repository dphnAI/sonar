# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification


@pytest.mark.parametrize(
    "model",
    ["nie3e/sentiment-polish-gpt2-small"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for head_dtype_str in ["float32", "model"]:
        with aphrodite_runner(
            model,
            max_model_len=512,
            dtype=dtype,
            hf_overrides={"head_dtype": head_dtype_str},
        ) as aphrodite_model:
            model_config = aphrodite_model.llm.llm_engine.model_config
            model_dtype = model_config.dtype
            head_dtype = model_config.head_dtype

            if head_dtype_str == "float32":
                assert head_dtype == torch.float32
            elif head_dtype_str == "model":
                assert head_dtype == model_dtype

            aphrodite_outputs = aphrodite_model.classify(example_prompts)

        for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
            hf_output = torch.tensor(hf_output).float()
            aphrodite_output = torch.tensor(aphrodite_output).float()

            assert torch.allclose(hf_output, aphrodite_output, atol=1e-2)
