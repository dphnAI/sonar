# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification


@pytest.mark.parametrize(
    "model",
    ["vllm-project/multi-label-class-classification-on-github-issues"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with sonar_runner(model, max_model_len=512, dtype=dtype) as sonar_model:
        sonar_outputs = sonar_model.classify(example_prompts)

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
        hf_output = torch.tensor(hf_output)
        sonar_output = torch.tensor(sonar_output)

        assert torch.allclose(hf_output, sonar_output, 1e-3 if dtype == "float" else 1e-2)
