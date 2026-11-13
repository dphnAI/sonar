# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch
from torch import fx as fx
from torch import nn

# This import automatically registers `torch.ops.silly.attention`
import tests.compile.silly_attention  # noqa
from sonar.compilation.counter import compilation_counter
from sonar.compilation.decorators import support_torch_compile
from sonar.compilation.passes.inductor_pass import (
    InductorPass,
    get_pass_context,
)
from sonar.config import (
    SonarConfig,
    set_current_sonar_config,
)
from sonar.config.compilation import CompilationConfig, CompilationMode
from sonar.config.scheduler import SchedulerConfig
from sonar.config.utils import Range
from sonar.forward_context import set_forward_context
from sonar.platforms import current_platform

BATCH_SIZE = 64
MLP_SIZE = 128

DEVICE_TYPE = current_platform.device_type


@pytest.fixture(autouse=True, scope="module")
def reset_default_device():
    """Tests here set the default device and leave it set, as they always
    have, so later tests in this file run on it. Reset it when the file ends,
    so other files run in the same pytest start from the default device."""
    yield
    torch.set_default_device(None)


@support_torch_compile
class TestModel(nn.Module):
    def __init__(self, *, sonar_config: SonarConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        x = attn_output
        x = x * 3
        return x


@torch.inference_mode
def run_model(sonar_config: SonarConfig, model: nn.Module, batch_sizes: list[int]):
    with set_forward_context({}, sonar_config=sonar_config):
        model(torch.randn(BATCH_SIZE, MLP_SIZE))
        for batch_size in batch_sizes:
            model(torch.randn(batch_size, MLP_SIZE))


class PostGradRangeChecker(InductorPass):
    def __init__(self, ranges: list[Range]):
        self.ranges = ranges
        self.num_calls = 0

    def __call__(self, graph: fx.Graph):
        compile_range = get_pass_context().compile_range
        assert compile_range in self.ranges, f"Compile range {compile_range} not in {self.ranges}"
        self.num_calls += 1

    def uuid(self) -> str:
        state: dict[str, Any] = {}
        return InductorPass.hash_dict(state)


def test_compile_ranges(disable_sonar_compile_cache):
    post_grad_range_checker = PostGradRangeChecker(
        [
            Range(start=1, end=8),
            Range(start=16, end=16),
            Range(start=9, end=32),
            Range(start=64, end=64),
            Range(start=128, end=128),
            Range(start=33, end=8192),
        ]
    )
    torch.set_default_device(DEVICE_TYPE)
    sonar_config = SonarConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.SONAR_COMPILE,
            compile_ranges_endpoints=[8, 32],
            compile_sizes=[16, 64, 128],
            inductor_compile_config={
                "post_grad_custom_post_pass": post_grad_range_checker,
            },
        ),
    )

    with set_current_sonar_config(sonar_config):
        model = TestModel(sonar_config=sonar_config, prefix="").eval()
        # Number of compilations: 3 compile ranges + 3 compile sizes
        batch_sizes = [1, 4, 16, 24, 48, 64, 8192]

        with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=1,
            num_backend_compilations=6,
        ):
            run_model(sonar_config, model, batch_sizes)
        assert post_grad_range_checker.num_calls == 6


def test_compile_config_get_compile_ranges():
    compilation_config = CompilationConfig(
        compile_ranges_endpoints=[8, 32],
    )
    SonarConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=compilation_config,
    )
    assert compilation_config.get_compile_ranges() == [
        Range(start=1, end=8),
        Range(start=9, end=32),
        Range(start=33, end=8192),
    ]


class PostGradStaticShapeChecker(InductorPass):
    """Asserts that compile_sizes entries produce graphs with fully concrete
    (non-symbolic) shapes, and compile_ranges entries have symbolic shapes."""

    def __init__(self):
        self.num_static_calls = 0
        self.num_dynamic_calls = 0

    def __call__(self, graph: fx.Graph):
        from torch.fx.experimental.symbolic_shapes import is_symbolic

        compile_range = get_pass_context().compile_range
        is_single = compile_range.is_single_size()

        for node in graph.nodes:
            val = node.meta.get("val")
            if val is None:
                val = node.meta.get("example_value")
            if isinstance(val, torch.Tensor):
                has_symbolic = any(is_symbolic(d) for d in val.shape)
                if is_single:
                    assert not has_symbolic, (
                        f"compile_sizes entry {compile_range}: node '{node.name}' has symbolic shape {val.shape}"
                    )
                else:
                    # compile_ranges should have at least some
                    # symbolic shapes (the batch dimension)
                    if has_symbolic:
                        self.num_dynamic_calls += 1
                        return

        if is_single:
            self.num_static_calls += 1

    def uuid(self) -> str:
        state: dict[str, Any] = {}
        return InductorPass.hash_dict(state)


def test_compile_sizes_produce_static_shapes(disable_sonar_compile_cache):
    """Verify that compile_sizes entries are compiled with fully concrete
    shapes (no SymInts), while compile_ranges entries retain dynamic shapes."""
    checker = PostGradStaticShapeChecker()
    torch.set_default_device(DEVICE_TYPE)
    sonar_config = SonarConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.SONAR_COMPILE,
            compile_ranges_endpoints=[8],
            compile_sizes=[16],
            inductor_compile_config={
                "post_grad_custom_post_pass": checker,
            },
        ),
    )

    with set_current_sonar_config(sonar_config):
        model = TestModel(sonar_config=sonar_config, prefix="").eval()
        # 3 compilations: Range(1,8), Range(9,8192), single-size 16
        with compilation_counter.expect(
            num_graphs_seen=1,
            num_piecewise_graphs_seen=1,
            num_backend_compilations=3,
        ):
            run_model(sonar_config, model, [1, 16, 64])

    # compile_sizes=16 should produce static shapes
    assert checker.num_static_calls == 1, f"Expected 1 static compilation, got {checker.num_static_calls}"
    # compile_ranges should produce dynamic shapes
    assert checker.num_dynamic_calls == 2, f"Expected 2 dynamic compilations, got {checker.num_dynamic_calls}"


def test_inductor_cache_compile_ranges(disable_sonar_compile_cache):
    # disable_sonar_compile_cache sets SONAR_DISABLE_COMPILE_CACHE=1 to
    # force multiple compilations by disabling Sonar's on-disk compile cache.
    post_grad_range_checker = PostGradRangeChecker(
        ranges=[
            Range(start=1, end=8),
            Range(start=9, end=8192),
        ]
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=8192,
        max_model_len=8192,
        is_encoder_decoder=False,
    )
    torch.set_default_device(DEVICE_TYPE)

    def create_sonar_config():
        return SonarConfig(
            scheduler_config=scheduler_config,
            compilation_config=CompilationConfig(
                mode=CompilationMode.SONAR_COMPILE,
                compile_ranges_endpoints=[8],
                inductor_compile_config={
                    "post_grad_custom_post_pass": post_grad_range_checker,
                },
            ),
        )

    sonar_config_1 = create_sonar_config()
    with set_current_sonar_config(sonar_config_1):
        model1 = TestModel(sonar_config=sonar_config_1, prefix="").eval()
        batch_sizes = [1, 16]
        run_model(sonar_config_1, model1, batch_sizes)
        assert post_grad_range_checker.num_calls == 2

    post_grad_range_checker.num_calls = 0
    # Create a new sonar config with the new pass context
    sonar_config_2 = create_sonar_config()
    with set_current_sonar_config(sonar_config_2):
        model2 = TestModel(sonar_config=sonar_config_2, prefix="").eval()
        batch_sizes = [4, 32]
        run_model(sonar_config_2, model2, batch_sizes)
        # Check that cache is used, so the number of calls
        # should be 0
        assert post_grad_range_checker.num_calls == 0
