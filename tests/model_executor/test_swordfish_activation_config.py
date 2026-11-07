# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sonar.model_executor.layers.fused_moe.activation import MoEActivation
from sonar.model_executor.layers.fused_moe.experts.swordfish_moe import SwordfishExperts


@pytest.mark.parametrize("grouped", [False, True])
def test_swordfish_forwards_activation_config(grouped):
    moe_config = SimpleNamespace(
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    quant_config = SimpleNamespace(
        use_int4_w4a16=True,
        w1_zp=None,
        w2_zp=None,
        w1_bias=None,
        w2_bias=None,
        gemm1_clamp_limit=7.0,
        gemm1_alpha=1.5,
        gemm1_beta=0.5,
        block_shape=[0, 128],
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
    )
    experts = SwordfishExperts(moe_config, quant_config)
    hidden = torch.ones((1, 128), dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    w1 = torch.empty((1, 4, 2, 1))
    w2 = torch.empty((1, 2, 2, 1))
    ids = torch.zeros((1, 1), dtype=torch.int32)
    weights = torch.ones((1, 1))
    activation_path = "sonar.model_executor.layers.fused_moe.modular_kernel.apply_moe_activation"
    ops_path = "sonar.model_executor.layers.fused_moe.experts.swordfish_moe"
    with (
        patch(activation_path) as activation,
        patch(f"{ops_path}.ops.swordfish_moe_mm", side_effect=[torch.ones((1, 256)), hidden]),
        patch(f"{ops_path}.ops.swordfish_prefill_mm", side_effect=[torch.ones((1, 256)), hidden]),
        patch(f"{ops_path}.ops.moe_sum"),
        patch(f"{ops_path}.moe_align_block_size", return_value=(ids, ids, ids)),
    ):
        if grouped:
            experts._apply_grouped(
                output,
                hidden,
                w1,
                w2,
                weights,
                ids,
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                1,
                128,
                128,
                1,
                False,
            )
        else:
            experts.apply(
                output,
                hidden,
                w1,
                w2,
                weights,
                ids,
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                1,
                None,
                None,
                None,
                torch.empty(0),
                torch.empty(0),
                None,
                False,
            )
    activation.assert_called_once()
    config = activation.call_args.kwargs["activation_config"]
    assert config.clamp_limit == 7.0
    assert config.alpha == 1.5
    assert config.beta == 0.5
