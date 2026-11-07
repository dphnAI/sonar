# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification

from sonar.platforms import current_platform
from sonar.utils.torch_utils import set_random_seed
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import softmax


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed all random number generators for reproducibility."""
    seed = 0
    set_random_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield


@pytest.mark.parametrize(
    "model",
    [
        "boltuix/NeuroBERT-NER",
    ],
)
# The float32 is required for this tiny model to pass the test.
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.core_model
@torch.inference_mode
def test_bert_model_runner_v2(
    hf_runner,
    sonar_runner,
    example_prompts,
    monkeypatch,
    model: str,
    dtype: str,
) -> None:
    prompt_batches = [[example_prompts[0]], example_prompts]

    monkeypatch.setenv("SONAR_USE_V2_MODEL_RUNNER", "1")
    with sonar_runner(model, max_model_len=None, dtype=dtype) as sonar_model:
        assert sonar_model.llm.llm_engine.sonar_config.use_v2_model_runner
        sonar_output_batches = [sonar_model.token_classify(prompts) for prompts in prompt_batches]

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))
        hf_output_batches = [[hf_outputs[0]], hf_outputs]

    # check logits difference
    for hf_outputs, sonar_outputs in zip(hf_output_batches, sonar_output_batches):
        for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
            hf_output = hf_output.detach().clone().cpu().float()
            sonar_output = sonar_output.detach().clone().cpu().float()
            torch.testing.assert_close(hf_output, sonar_output, atol=3.2e-2, rtol=1e-3)


@pytest.mark.parametrize("model", ["disham993/electrical-ner-ModernBERT-base"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.flaky(reruns=3)
@torch.inference_mode
def test_modernbert_models(
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    # NOTE: https://github.com/vllm-project/vllm/pull/32403
    # `disham993/electrical-ner-ModernBERT-base` is a randomly initialized
    # model, which can cause numerical precision variance and edge cases.
    # We use @flaky(reruns=3) to mitigate intermittent failures.
    print(
        f"\n[NOTE] Testing {model} (randomly initialized weights) - "
        "flaky tolerance enabled due to numerical precision variance."
    )

    with sonar_runner(model, max_model_len=None, dtype=dtype) as sonar_model:
        sonar_outputs = sonar_model.token_classify(example_prompts)

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        sonar_output = sonar_output.detach().clone().cpu().float()
        torch.testing.assert_close(hf_output, sonar_output, atol=6e-2, rtol=1e-3)


@pytest.mark.parametrize("model", ["Davlan/xlm-roberta-base-ner-hrl"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.flaky(reruns=3)
@torch.inference_mode
def test_xlm_roberta_models(
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with sonar_runner(model, max_model_len=None, dtype=dtype) as sonar_model:
        sonar_outputs = sonar_model.token_classify(example_prompts)

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        sonar_output = sonar_output.detach().clone().cpu().float()
        torch.testing.assert_close(hf_output, sonar_output, atol=3.2e-2, rtol=1e-3)


PRIVACY_FILTER_PROMPTS = [
    "My name is Harry Potter.",
    "Email me at harry.potter@hogwarts.edu.",
    "Call me on +44 20 7946 0958 tomorrow.",
    "My account number is 12345678 and the API key is sk-live-abc123def456.",
    "I live at 4 Privet Drive, Little Whinging.",
    "Visit https://example.com/profile/harry for more info.",
    "We met on 12 January 2024.",
]


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="Workspace growth allocation issue on ROCm. See https://github.com/vllm-project/vllm/issues/48510",
)
@pytest.mark.parametrize("model", ["openai/privacy-filter"])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@torch.inference_mode
def test_openai_privacy_filter(
    hf_runner,
    sonar_runner,
    model: str,
    dtype: str,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_transformers_version(on_fail="skip")

    with sonar_runner(model, max_model_len=None, dtype=dtype) as sonar_model:
        sonar_outputs = sonar_model.token_classify(PRIVACY_FILTER_PROMPTS)

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in PRIVACY_FILTER_PROMPTS:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        sonar_output = sonar_output.detach().clone().cpu().float()
        torch.testing.assert_close(hf_output, sonar_output, atol=0.1, rtol=1e-2)


@pytest.mark.parametrize("model", ["bd2lcco/Qwen3-0.6B-finetuned"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.flaky(reruns=3)
@torch.inference_mode
def test_auto_conversion(
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with sonar_runner(model, max_model_len=1024, dtype=dtype) as sonar_model:
        sonar_outputs = sonar_model.token_classify(example_prompts)

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForTokenClassification) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        sonar_output = sonar_output.detach().clone().cpu().float()
        assert torch.allclose(hf_output, sonar_output, atol=1e-2)


@pytest.mark.parametrize(
    "model",
    [
        # Original Google checkpoint: legacy `gamma`/`beta` LayerNorm names, an
        # NSP head (`cls.seq_relationship.*`) and a decoder tied to the input
        # embeddings (no explicit decoder weight in the checkpoint).
        "google-bert/bert-base-uncased",
    ],
)
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_bert_for_masked_lm(
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    # BertForMaskedLM exposes its MLM head as a token-level pooling task; the
    # head applies softmax over the vocabulary, so each output row is a
    # distribution (matching HF's softmax(logits) below).
    with sonar_runner(model, max_model_len=None, dtype=dtype) as sonar_model:
        sonar_outputs = sonar_model.token_classify(example_prompts)

    # Run hf_runner reference with "highest" fp32 precision to match
    # default behavior of Sonar. This is needed on ROCm since the
    # pooling tests set matmul precision to "high" in conftest.py.
    prev_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with hf_runner(
            model,
            dtype=dtype,
            auto_cls=AutoModelForMaskedLM,
        ) as hf_model:
            tokenizer = hf_model.tokenizer
            hf_outputs = []
            for prompt in example_prompts:
                inputs = tokenizer([prompt], return_tensors="pt")
                inputs = hf_model.wrap_device(inputs)
                output = hf_model.model(**inputs)
                hf_outputs.append(softmax(output.logits[0]))
    finally:
        torch.set_float32_matmul_precision(prev_matmul_precision)

    # Compare the per-token vocabulary distributions position by position.
    for hf_output, sonar_output in zip(hf_outputs, sonar_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        sonar_output = sonar_output.detach().clone().cpu().float()
        assert hf_output.shape == sonar_output.shape
        assert torch.equal(hf_output.argmax(dim=-1), sonar_output.argmax(dim=-1))
        torch.testing.assert_close(hf_output, sonar_output, atol=3.2e-2, rtol=1e-3)
