# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from sonar.compilation.decorators import support_torch_compile
from sonar.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    SonarConfig,
    set_current_sonar_config,
)

# This import automatically registers `torch.ops.silly.attention`
from .. import silly_attention  # noqa: F401


@support_torch_compile
class SnapshotModel(nn.Module):
    """Returns a snapshot of its input, then mutates the input in place, like a
    model that forwards its input embeddings past the compile boundary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snapshot = x.clone()
        out = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, out)
        x.add_(out)
        return snapshot


@pytest.mark.parametrize("splitting_ops", [["silly::attention"], []], ids=["piecewise", "full_graph"])
@torch.inference_mode()
def test_snapshot_clone_survives_input_mutation(splitting_ops, disable_sonar_compile_cache):
    """Ensure a clone of the input keeps its value when the input is later mutated."""
    sonar_config = SonarConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.SONAR_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            splitting_ops=splitting_ops,
        )
    )
    with set_current_sonar_config(sonar_config):
        model = SnapshotModel()

    x = torch.randn(16, 8, device="cuda")
    expected = x.clone()
    snapshot = model(x)

    torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)
