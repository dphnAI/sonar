# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification


@pytest.mark.parametrize(
    "model",
    ["Rami/multi-label-class-classification-on-github-issues"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with aphrodite_runner(model, max_model_len=512, dtype=dtype) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.classify(example_prompts)

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for hf_output, aphrodite_output in zip(hf_outputs, aphrodite_outputs):
        hf_output = torch.tensor(hf_output)
        aphrodite_output = torch.tensor(aphrodite_output)

        assert torch.allclose(hf_output, aphrodite_output, 1e-3 if dtype == "float" else 1e-2)
