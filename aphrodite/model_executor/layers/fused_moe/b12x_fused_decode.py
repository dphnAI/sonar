# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x's compact W4A8 decode MoE in one SM120 kernel.

For up to eight tokens b12x runs rows-quant, FC1, the SwiGLU activation and FC2
as separate kernels. ``install`` wraps b12x's ``launch_w4a8_compact_micro`` so
the shapes ``csrc/moe/b12x_moe_fused_sm120.cu`` computes exactly like b12x run
b12x's rows-quant and then that persistent kernel, which streams FC1 and FC2
weights back to back. ``projections``, ``intermediate`` and ``route_output``
keep b12x's bits; every other launch is b12x's own.
"""

import functools
from typing import Any

import torch

# The kernel's control block needs 516 ints; the op rejects a smaller one.
_CTRL_INTS = 1024
_ctrl_blocks: dict[int, torch.Tensor] = {}


def _supported(
    *,
    n: int,
    k: int,
    num_topk: int,
    max_tokens: int,
    ids_dtype: torch.dtype,
    csf_inline: bool,
    experts: int,
    swiglu_limit: float | None,
) -> bool:
    """Shapes the kernel computes like b12x, and where it is faster.

    Below 32 pairs (the DSpark drafter's top-3 MoE) the FC1 phase has too few
    weight streams and b12x's kernels win.
    FC1 stages scales in 16-byte vectors, with sixteen vectors per row.
    Its seven unconditional K64 preloads also require at least 512 columns.
    """
    return (
        n == 576
        and 512 <= k <= 8192
        and k % 512 == 0
        and 1 <= max_tokens <= 8
        and 32 <= max_tokens * num_topk <= 64
        and ids_dtype == torch.int32
        and not csf_inline
        and 1 <= experts <= 384
        and swiglu_limit is not None
    )


def _ctrl_block(device: torch.device) -> torch.Tensor:
    index = device.index if device.index is not None else torch.accelerator.current_device_index()
    ctrl = _ctrl_blocks.get(index)
    if ctrl is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the fused b12x decode MoE must run once before CUDA graph capture")
        ctrl = torch.empty(_CTRL_INTS, dtype=torch.int32, device=torch.device("cuda", index))
        torch.ops._C.b12x_w4a8_moe_fused_ctrl_init(ctrl)
        _ctrl_blocks[index] = ctrl
    return ctrl


def _fused(micro: Any, kw: dict[str, Any]) -> torch.Tensor | None:
    """Run the fused path, or return None for b12x's launcher."""
    a, ids, weights, w2, alpha1 = kw["a"], kw["topk_ids"], kw["topk_weights"], kw["w2"], kw["alpha1"]
    cap, topk = int(kw["max_tokens"]), int(kw["num_topk"])
    if (
        a.dtype != torch.bfloat16
        or a.ndim != 2
        or not a.is_contiguous()
        or weights.dtype != torch.float32
        or not ids.is_contiguous()
        or not weights.is_contiguous()
        or kw.get("max_active_clusters") is not None
    ):
        return None
    k, tokens, experts = int(a.shape[1]), int(a.shape[0]), int(alpha1.numel())
    w2_bytes = w2.numel() * w2.element_size()
    if experts < 1 or k % 256 or (2 * w2_bytes) % (experts * k) or tokens > cap or ids.numel() < tokens * topk:
        return None
    n = (2 * w2_bytes) // (experts * k)
    if not _supported(
        n=n,
        k=k,
        num_topk=topk,
        max_tokens=cap,
        ids_dtype=ids.dtype,
        csf_inline=bool(kw.get("csf_inline", False)),
        experts=experts,
        swiglu_limit=kw["swiglu_limit"],
    ):
        return None
    layout = micro._layout(cap, topk, k, n)
    storage = micro._as_bytes(kw["scratch"])
    if storage.numel() < layout["total"][1]:
        return None

    def region(name: str) -> torch.Tensor:
        offset, size = layout[name]
        return storage.narrow(0, offset, size)

    values = region("a_values").view(cap, k)
    scale_rows = region("a_scales").view(cap, k // 32)
    scale_mma = region("a_mma_scales")
    route = region("route_output").view(torch.bfloat16).view(cap * topk, k)
    quantize = kw.get("_prepared_quantize")
    if quantize is None:
        micro.quantize_mxfp8_rows_cute(a, values, scale_rows, scale_mma, expected_m=cap)
    else:
        quantize(a, values, scale_rows, scale_mma)
    torch.ops._C.b12x_w4a8_moe_fused_sm120(
        values,
        scale_rows,
        kw["w13"],
        kw["w13_scales"],
        region("projections"),
        region("intermediate"),
        w2,
        kw["w2_scales"],
        ids.reshape(-1),
        weights.reshape(-1),
        alpha1,
        kw["input_scale"],
        kw["alpha2"],
        kw["down_scale"],
        route,
        _ctrl_block(a.device),
        tokens * topk,
        topk,
        cap * topk,
        n,
        float(kw["swiglu_limit"]),
    )
    return route.narrow(0, 0, tokens * topk)


def install() -> None:
    """Route b12x's eligible compact W4A8 decode launches to the fused kernel."""
    if not hasattr(torch.ops._C, "b12x_w4a8_moe_fused_sm120"):
        return
    from b12x.moe._shared.kernels import w4a8_compact_micro as micro

    original = micro.launch_w4a8_compact_micro
    if getattr(original, "_aphrodite_fused", False):
        return

    @functools.wraps(original)
    def launch(**kw: Any) -> torch.Tensor:
        route = _fused(micro, kw)
        return original(**kw) if route is None else route

    launch._aphrodite_fused = True  # type: ignore[attr-defined]
    micro.launch_w4a8_compact_micro = launch
