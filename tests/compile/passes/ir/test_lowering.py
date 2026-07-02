# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

import aphrodite.kernels  # noqa: F401 to register kernels
from aphrodite import ir
from aphrodite.compilation.passes.ir.lowering_pass import (
    AphroditeIRLoweringPass,
)
from aphrodite.config import get_current_vllm_config
from aphrodite.ir import ops
from aphrodite.platforms import current_platform

from ...backend import TestBackend


class Model(nn.Module):
    def __init__(self, hidden_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.rms_norm(x1, self.weight, 1e-5)
        x3 = x2 * 5.0
        # no weight
        x4 = ops.rms_norm(x3, None, 1e-5)
        x5 = x4 / 2.0
        # dispatch to native due to variance_size parameter
        x6 = ops.rms_norm(x5, self.weight, 1e-5, self.hidden_size // 2)
        return x6 + 3.0


@pytest.mark.parametrize("rms_provider", ops.rms_norm.supported_providers())
def test_lowering_rms_norm(rms_provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = AphroditeIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = Model()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    with (
        ops.rms_norm.set_priority([rms_provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["rms_norm"]
    assert len(selected) == 3
    assert selected["rms_norm"] == rms_provider
    assert selected["rms_norm_1"] == rms_provider
    assert selected["rms_norm_2"] == "native"

    # Compiled function guards on global value, avoid recompilation
    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    torch.testing.assert_close(output_unlowered, output)
    torch.testing.assert_close(output_unlowered, output2)
