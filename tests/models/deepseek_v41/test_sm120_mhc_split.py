# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split lagged mHC post_pre must match b12x's fused decode path bit for bit."""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

b12x_mhc = pytest.importorskip("b12x.norm.mhc._kernels")

from b12x._lib.compile_plan import compile_only_launches  # noqa: E402

from aphrodite.models.deepseek_v41.sm120.mhc_split import (  # noqa: E402
    PARTIALS,
    PARTIALS_PER_CTA,
    join_mhc_split,
    lagged_post_pre,
    prepare_mhc_split,
)

HIDDEN = 5120
RMS_EPS = 1e-20
HC_EPS = 1e-6
ITERATIONS = 20
_PROGRAMS = {}


def _run_b12x(launch, key, **kwargs):
    if key not in _PROGRAMS:
        with compile_only_launches():
            _PROGRAMS[key] = launch(**kwargs)
    launch(**kwargs, _prepared=_PROGRAMS[key])


def _fused(inputs, out, y, post, comb, pre_out):
    rows = inputs["x"].shape[0]
    split_k = b12x_mhc._split_k_for_hidden(HIDDEN)
    partials = torch.empty((rows, split_k, PARTIALS), dtype=torch.float32, device=y.device)
    native = types.SimpleNamespace(
        source_splits=0, decode_tile_n=0, decode_bf16x2=False, partials_per_cta=PARTIALS_PER_CTA
    )
    _run_b12x(
        b12x_mhc._run_mhc_post_pre_partial_launch,
        ("partial", rows),
        x=inputs["x"],
        residual=inputs["residual"],
        prev_post=inputs["prev_post"],
        prev_comb=inputs["prev_comb"],
        fn=inputs["fn"],
        partials=partials,
        out=out,
        pre_mix=inputs["pre_mix"],
        y=y,
        native=native,
    )
    _run_b12x(
        b12x_mhc._run_mhc_finalize_gram_launch,
        ("finalize", rows),
        residual=out,
        partials=partials,
        scale=inputs["scale"],
        bias=inputs["bias"],
        y=y,
        post=post,
        comb=comb,
        rms_eps=RMS_EPS,
        hc_eps=HC_EPS,
        sinkhorn_iters=ITERATIONS,
        norm_weight=inputs["weight"],
        norm_eps=RMS_EPS,
        fuse_norm=True,
        pre_mix=inputs["pre_mix"],
        pre_out=pre_out,
        lagged_prepared=True,
    )


def _outputs(rows, device):
    return (
        torch.full((rows, 4, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device),
        torch.full((rows, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device),
        torch.full((rows, 4), float("nan"), device=device),
        torch.full((rows, 4, 4), float("nan"), device=device),
        torch.full((rows, 4), float("nan"), device=device),
    )


@pytest.mark.parametrize("rows", [1, 7, 8, 64])
def test_split_matches_fused_lagged_post_pre(rows):
    device = torch.device("cuda", torch.cuda.current_device())
    prepare_mhc_split(device, HIDDEN, RMS_EPS, HC_EPS, ITERATIONS)
    gen = torch.Generator(device="cuda").manual_seed(rows)
    inputs = {
        "x": (torch.randn(rows, HIDDEN, generator=gen, device=device) * 4).to(torch.bfloat16),
        "residual": (torch.randn(rows, 4, HIDDEN, generator=gen, device=device) * 4).to(torch.bfloat16),
        "prev_post": torch.rand(rows, 4, generator=gen, device=device) * 2,
        "prev_comb": torch.softmax(torch.randn(rows, 4, 4, generator=gen, device=device), dim=-1),
        "fn": torch.randn(24, 4 * HIDDEN, generator=gen, device=device) * 0.02,
        "scale": torch.rand(3, generator=gen, device=device) + 0.5,
        "bias": torch.randn(24, generator=gen, device=device),
        "weight": (torch.rand(HIDDEN, generator=gen, device=device) + 0.5).to(torch.bfloat16),
        "pre_mix": torch.softmax(torch.randn(rows, 4, generator=gen, device=device), dim=-1),
    }
    expected = _outputs(rows, device)
    _fused(inputs, *expected)

    actual = _outputs(rows, device)
    lagged_post_pre(
        inputs["x"],
        inputs["residual"],
        inputs["prev_post"],
        inputs["prev_comb"],
        inputs["fn"],
        inputs["scale"],
        inputs["bias"],
        inputs["weight"],
        inputs["pre_mix"],
        *actual,
    )
    join_mhc_split(device)
    torch.accelerator.current_stream().synchronize()

    for name, got, want in zip(("residual_out", "y", "post", "comb", "pre_out"), actual, expected):
        assert torch.equal(
            got.view(torch.int16 if got.dtype == torch.bfloat16 else torch.int32),
            want.view(torch.int16 if want.dtype == torch.bfloat16 else torch.int32),
        ), name


def test_broadcast_projection_refit_preserves_captured_buffer():
    from aphrodite.models.deepseek_v41.sm120.model import DeepseekV4DecoderLayer

    layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_mult = 4
    layer.hidden_size = 8
    layer.hc_attn_fn = torch.nn.Parameter(torch.randn(24, 32, dtype=torch.float32, device="cuda"), requires_grad=False)
    layer.hc_attn_fn_broadcast = None
    layer.finalize_mhc_broadcast_weights()
    buffer = layer.hc_attn_fn_broadcast
    expected = layer.hc_attn_fn.view(24, 4, 8).sum(dim=1)
    torch.testing.assert_close(buffer, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = buffer.clone()
    layer.hc_attn_fn.add_(1)
    layer.finalize_mhc_broadcast_weights()
    graph.replay()

    assert layer.hc_attn_fn_broadcast is buffer
    expected = layer.hc_attn_fn.view(24, 4, 8).sum(dim=1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
