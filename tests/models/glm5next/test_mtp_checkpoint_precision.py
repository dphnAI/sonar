# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Respect BF16 MTP weights in mixed ModelOpt NVFP4 checkpoints."""

from types import SimpleNamespace

import pytest

from sonar.models.glm5next.common import mtp


@pytest.mark.parametrize("scale_suffix", [None, "weight_scale", "weight_scale_2", "input_scale"])
def test_mtp_precision_uses_only_selected_layer_scales(monkeypatch, scale_suffix):
    weights = {
        "model.layers.31.mlp.experts.0.gate_proj.weight": "base.safetensors",
        "model.layers.31.mlp.experts.0.gate_proj.weight_scale": "base.safetensors",
        "model.layers.32.mlp.experts.0.gate_proj.weight": "mtp.safetensors",
    }
    if scale_suffix is not None:
        weights[f"model.layers.32.mlp.experts.0.gate_proj.{scale_suffix}"] = "mtp.safetensors"
    monkeypatch.setattr(mtp, "get_hf_file_to_dict", lambda *args: {"weight_map": weights})
    quant = SimpleNamespace(get_name=lambda: "modelopt_fp4")
    assert mtp.checkpoint_stores_layer_unquantized(quant, "checkpoint", None, 32) == (scale_suffix is None)
    # An absent layer is not evidence that it is unquantized.
    assert not mtp.checkpoint_stores_layer_unquantized(quant, "checkpoint", None, 33)


@pytest.mark.parametrize("index", [None, {}, {"weight_map": {}}])
def test_mtp_precision_does_not_guess_without_weight_inventory(monkeypatch, index):
    monkeypatch.setattr(mtp, "get_hf_file_to_dict", lambda *args: index)
    quant = SimpleNamespace(get_name=lambda: "modelopt_fp4")
    assert not mtp.checkpoint_stores_layer_unquantized(quant, "checkpoint", None, 32)


def test_other_quantization_methods_do_not_read_nvfp4_inventory(monkeypatch):
    def unexpected_lookup(*args):
        raise AssertionError("Only ModelOpt FP4 needs inventory-based MTP precision")

    monkeypatch.setattr(mtp, "get_hf_file_to_dict", unexpected_lookup)
    assert not mtp.checkpoint_stores_layer_unquantized(None, "checkpoint", None, 32)
    quant = SimpleNamespace(get_name=lambda: "fp8")
    assert not mtp.checkpoint_stores_layer_unquantized(quant, "checkpoint", None, 32)
