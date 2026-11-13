# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--mamba-ssm-cache-dtype` must reach GLM-5.3-Flash's KDA recurrent state
(it used to be dropped, leaving the state float32 unconditionally)."""

from types import SimpleNamespace

import pytest
import torch

from sonar.models.glm5next.common.kda import Glm5NextLinearAttention
from sonar.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)


@pytest.mark.parametrize(
    ("ssm_dtype", "expected"),
    [("auto", torch.float32), ("float32", torch.float32), ("bfloat16", torch.bfloat16)],
)
def test_kda_recurrent_state_dtype_follows_ssm_cache_dtype(ssm_dtype, expected):
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype=ssm_dtype,
        use_kda_recoverssm=False,
    )
    sonar_config = SimpleNamespace(model_config=model_config, cache_config=cache_config)
    want = (torch.bfloat16, expected)

    assert Glm5NextForCausalLM.get_mamba_state_dtype_from_config(sonar_config) == want
    assert Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(sonar_config) == want
    # The layer's own dtype must agree with the model-level one, since the
    # KV-cache spec and the kernels read them separately.
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    layer.model_config, layer.cache_config = model_config, cache_config
    assert layer.get_state_dtype() == want


@pytest.mark.parametrize("compact", [False, True])
def test_compact_spec_uses_one_recurrent_state_and_keeps_conv_history(compact):
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="float32",
        use_kda_recoverssm=compact,
        use_replayssm=False,
        mamba_block_size=256,
        mamba_page_size_padded=None,
        mamba_cache_mode="align",
    )
    config = SimpleNamespace(model_config=model_config, cache_config=cache_config, num_speculative_tokens=3)
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    layer.model_config, layer.cache_config = model_config, cache_config
    layer.tp_size, layer.num_heads, layer.head_dim = 1, 4, 128
    layer.conv_size, layer.num_spec = 4, 3
    layer.kda_prefill_backend = "flashkda"
    spec = layer.get_kv_cache_spec(config)
    assert spec.num_speculative_blocks == (0 if compact else 3)
    assert spec.shapes == layer.get_state_shape()
    assert 6 in spec.shapes[0]  # Three history tokens plus three drafts.
    assert spec.num_prefill_checkpoint_blocks == 1

    if compact:
        layer.supports_compact_spec_state = False
        with pytest.raises(ValueError, match="does not support compact"):
            layer.get_kv_cache_spec(config)
