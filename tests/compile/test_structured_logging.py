# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import regex as re
import torch
from torch import nn

import tests.compile.silly_attention  # noqa
from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.config import AphroditeConfig, set_current_aphrodite_config
from aphrodite.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)
from aphrodite.config.scheduler import SchedulerConfig
from aphrodite.forward_context import set_forward_context
from aphrodite.platforms import current_platform

MLP_SIZE = 64
DEVICE_TYPE = current_platform.device_type


@support_torch_compile
class SimpleModel(nn.Module):
    """A simple model with a splitting op for piecewise compilation."""

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "", **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output * 2
        return x


class TraceStructuredCapture:
    """Captures trace_structured calls for testing."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, event_type: str, metadata_fn=None, payload_fn=None, **kwargs):
        """Capture a trace_structured call."""
        metadata = metadata_fn() if metadata_fn else {}
        self.calls.append(
            {
                "event_type": event_type,
                "metadata": metadata,
            }
        )

    def get(self, event_type: str, name_pattern: str) -> list[dict]:
        """Get all calls with the given event type and name matching pattern.

        Args:
            event_type: The event type to filter by (e.g., "artifact", "graph_dump")
            name_pattern: Regex pattern to match against the artifact name
        """
        regex = re.compile(name_pattern)
        return [
            c
            for c in self.calls
            if c["event_type"] == event_type and regex.fullmatch(c.get("metadata", {}).get("name", ""))
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_aphrodite_structured_logging_artifacts(use_fresh_inductor_cache):
    """Test that all expected Aphrodite artifacts are logged during compilation."""
    torch.set_default_device(DEVICE_TYPE)

    capture = TraceStructuredCapture()

    aphrodite_config = AphroditeConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.APHRODITE_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            compile_sizes=[8],
            splitting_ops=["silly::attention"],
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
    )

    # Patch trace_structured to capture calls
    with (
        patch("aphrodite.compilation.backends.trace_structured", capture),
        patch("aphrodite.compilation.piecewise_backend.trace_structured", capture),
        set_current_aphrodite_config(aphrodite_config),
    ):
        model = SimpleModel(aphrodite_config=aphrodite_config, prefix="test")
        with set_forward_context({}, aphrodite_config=aphrodite_config):
            model(torch.randn(8, MLP_SIZE))

    config_artifacts = capture.get("artifact", "aphrodite_compilation_config")
    assert len(config_artifacts) == 1, f"Expected 1 aphrodite_compilation_config, got {len(config_artifacts)}"
    aphrodite_piecewise_split_graph = capture.get("graph_dump", "aphrodite_piecewise_split_graph")
    assert len(aphrodite_piecewise_split_graph) == 1, (
        f"Expected 1 toplevel piecewise split graph, got {len(aphrodite_piecewise_split_graph)}"
    )
    compile_start_artifacts = capture.get("artifact", "aphrodite_piecewise_compile_start")
    assert len(compile_start_artifacts) == 4, (
        "Expected 4 aphrodite_piecewise_compile_start "
        "(2 subgraphs x 2 ranges each: dynamic + compile size), "
        f"got {len(compile_start_artifacts)}"
    )
    submod_dumps = capture.get("graph_dump", r"aphrodite_submod_.*")
    assert len(submod_dumps) == 2, (
        f"Expected 2 submods (one before attention, one after attention), got {len(submod_dumps)}"
    )
