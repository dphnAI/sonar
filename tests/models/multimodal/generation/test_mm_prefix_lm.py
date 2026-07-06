# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch
from transformers import AutoModelForImageTextToText

from aphrodite.platforms import current_platform

from ....conftest import AphroditeRunner, HfRunner, ImageTestAssets
from .vlm_utils import model_utils

MODEL = "google/gemma-3-4b-it"
PROMPT = (
    "<bos><start_of_turn>user\n"
    "<start_of_image>What is the content in the center of the image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)


def _install_prefill_hidden_capture(model):
    model = getattr(model, "module", model)
    model._prefill_hidden = None

    language_model = model.language_model.model
    original_forward = language_model.forward

    def forward(*args, **kwargs):
        hidden_states = original_forward(*args, **kwargs)
        if model._prefill_hidden is None and torch.is_tensor(hidden_states):
            model._prefill_hidden = hidden_states.detach().float().cpu()
        return hidden_states

    language_model.forward = forward


def _get_prefill_hidden(model):
    model = getattr(model, "module", model)
    hidden = getattr(model, "_prefill_hidden", None)
    assert hidden is not None
    return hidden


def _get_hf_prefill_hidden(hf_model: HfRunner, image: Any):
    inputs = hf_model.get_inputs([PROMPT], images=[image])[0]
    with torch.no_grad():
        outputs = hf_model.model.model(
            **hf_model.wrap_device(inputs),
            use_cache=False,
        )
    return outputs.last_hidden_state[0].detach().float().cpu()


def _get_aphrodite_prefill_hidden(
    aphrodite_runner: type[AphroditeRunner],
    image: Any,
    aphrodite_runner_kwargs: dict[str, Any],
):
    with aphrodite_runner(
        MODEL,
        max_model_len=4096,
        max_num_seqs=2,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        **aphrodite_runner_kwargs,
    ) as aphrodite_model:
        aphrodite_model.apply_model(_install_prefill_hidden_capture)
        aphrodite_model.generate_greedy([PROMPT], max_tokens=1, images=[image])
        return aphrodite_model.apply_model(_get_prefill_hidden)[0]


@pytest.mark.core_model
@pytest.mark.skipif(current_platform.is_rocm(), reason="ROCm attention has accuracy issue for this test")
def test_mm_prefix_lm_e2e(
    hf_runner: type[HfRunner],
    aphrodite_runner: type[AphroditeRunner],
    image_assets: ImageTestAssets,
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression: Gemma3 native prefill must apply image prefix-LM mask."""
    monkeypatch.setenv("APHRODITE_ALLOW_INSECURE_SERIALIZATION", "1")
    image = image_assets[0].pil_image

    aphrodite_runner_kwargs: dict[str, Any] = {
        "mm_processor_cache_gb": 0,
        "mm_processor_kwargs": {"do_pan_and_scan": True},
    }
    aphrodite_hidden = _get_aphrodite_prefill_hidden(
        aphrodite_runner,
        image,
        aphrodite_runner_kwargs,
    )

    hf_model = hf_runner(
        MODEL,
        auto_cls=AutoModelForImageTextToText,
    )
    hf_model = model_utils.gemma3_patch_hf_runner(hf_model)

    with hf_model:
        hf_hidden = _get_hf_prefill_hidden(hf_model, image)

    assert aphrodite_hidden.shape == hf_hidden.shape

    full_cos = torch.nn.functional.cosine_similarity(aphrodite_hidden.flatten(), hf_hidden.flatten(), dim=0)
    image_cos = torch.nn.functional.cosine_similarity(
        aphrodite_hidden[1:769].flatten(), hf_hidden[1:769].flatten(), dim=0
    )

    assert full_cos > 0.9, f"Gemma3 mm-prefix-LM full prefill hidden states should be close to HF; got {full_cos=}"
    assert image_cos > 0.9, f"Gemma3 mm-prefix-LM image prefill hidden states should be close to HF; got {image_cos=}"
