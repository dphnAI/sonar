# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm L2 with weights the next sublayer reads, during the all-reduce before it.

At decode sizes an all-reduce mostly waits on peers while the GPU's memory
system sits idle, and the sublayer after it then reads its weights cold from
HBM. A side-stream read of those weights during the all-reduce lets them hit
L2 instead: the next layer's attention projections during the MoE all-reduce,
and the router during the attention all-reduce. It only reads static weights,
so it cannot change results; callers join the side stream once per forward.
"""

import torch

from aphrodite.triton_utils import tl, triton

CTAS = 96
BLOCK = 2048


@triton.jit
def _prefetch_kernel(Bases, Words, Out, SEGMENTS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for segment in tl.static_range(SEGMENTS):
        base = tl.load(Bases + segment).to(tl.pointer_type(tl.int32))
        words = tl.load(Words + segment)
        for start in range(pid * BLOCK, words, step):
            offs = start + tl.arange(0, BLOCK)
            acc += tl.load(base + offs, mask=offs < words, other=0)
    # The store keeps the loads live; the value is never read.
    tl.store(Out + pid, tl.sum(acc, 0))


def _span(tensor: torch.Tensor) -> tuple[int, int]:
    """4-byte aligned [start, end) of the bytes a strided tensor addresses."""
    extent = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    start = tensor.data_ptr()
    end = start + extent * tensor.element_size()
    return start & ~3, (end + 3) & ~3


class PrefetchPlan:
    def __init__(self, tensors: list[torch.Tensor]):
        spans = sorted({_span(tensor) for tensor in tensors})
        device = tensors[0].device
        self.nbytes = sum(end - start for start, end in spans)
        self.segments = len(spans)
        self.bases = torch.tensor([start for start, _ in spans], dtype=torch.int64, device=device)
        self.words = torch.tensor([(end - start) // 4 for start, end in spans], dtype=torch.int64, device=device)
        self.sink = torch.empty(CTAS, dtype=torch.int32, device=device)


def _rows(rows) -> tuple[torch.Tensor, torch.Tensor]:
    # Decode GEMMs read the tiled copy when one is packed.
    return (rows.values_tiled if rows.values_tiled is not None else rows.values, rows.scale_mma)


def attention_plan(attn) -> PrefetchPlan | None:
    """Weights the attention projections read, in their order of use."""
    tensors: list[torch.Tensor] = []
    linears = [attn.fused_wqa_wkv, attn.wq_b]
    if getattr(attn, "indexer", None) is not None:
        linears.append(attn.indexer.wq_b)
    for linear in linears:
        packed = getattr(linear, "b12x_weight", None)
        if packed is not None:
            tensors.extend(_rows(packed.weight))
    wo = getattr(attn, "_wo_projection_weights", None)
    if wo is not None:
        # Decode wo_b quantizes its input inside the GEMM loop, which bounds
        # it whether or not its weights are in L2, so only wo_a is warmed.
        tensors.extend(_rows(wo.wo_a))
    return PrefetchPlan(tensors) if tensors else None


def router_plan(ffn) -> PrefetchPlan | None:
    """The router weight the MoE reads first after the attention all-reduce."""
    weight = getattr(getattr(ffn, "gate", None), "weight", None)
    return PrefetchPlan([weight]) if weight is not None else None


_STREAMS: dict[int, torch.cuda.Stream] = {}
_PENDING: set[int] = set()


def launch(plan: PrefetchPlan) -> None:
    """Start reading `plan` on a side stream ordered after the current stream."""
    device = plan.bases.device
    stream = _STREAMS.get(device.index)
    if stream is None:
        stream = _STREAMS[device.index] = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        _prefetch_kernel[(CTAS,)](plan.bases, plan.words, plan.sink, plan.segments, BLOCK, num_warps=4)
    _PENDING.add(device.index)


def join(device: torch.device) -> None:
    """Order the current stream after any pending prefetch (required before capture ends)."""
    if device.index in _PENDING:
        torch.cuda.current_stream(device).wait_stream(_STREAMS[device.index])
        _PENDING.discard(device.index)
