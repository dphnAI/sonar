# SPDX-License-Identifier: Apache-2.0
"""The fused b12x W4A8 decode MoE (csrc/moe/b12x_moe_fused_sm120.cu) against b12x's own kernels, bit for bit."""

import pytest
import torch

from aphrodite.model_executor.layers.fused_moe import b12x_fused_decode
from aphrodite.platforms import current_platform
from aphrodite.utils.b12x import get_b12x_fused_moe

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(120)
    or not hasattr(torch.ops._C, "b12x_w4a8_moe_fused_sm120")
    or get_b12x_fused_moe() is None,
    reason="requires SM120, the fused decode MoE op and b12x",
)

# One TP4 rank of DeepSeek-V4.1-Flash: 384 experts, hidden 5120, intermediate 576, top-6, a DSpark k=7 verify step.
E, H, N, TOPK, LIMIT, TOKENS = 384, 5120, 576, 6, 10.0, 8


@pytest.fixture(scope="module")
def moe():
    import b12x.moe.fused_moe as fm
    from b12x.moe._shared.kernels import w4a8_compact_micro as micro
    from b12x.preparation import PreparedCall, prepare_default

    b12x_fused_decode.install()
    fused = micro.launch_w4a8_compact_micro
    original = fused.__wrapped__
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randint(0, 256, (E, 2 * N, H // 2), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (E, H, N // 2), dtype=torch.uint8, device="cuda", generator=g)
    w1s = torch.randint(119, 123, (E, 2 * N, H // 32), dtype=torch.uint8, device="cuda", generator=g)
    w2s = torch.randint(119, 123, (E, H, N // 32), dtype=torch.uint8, device="cuda", generator=g)
    for packed in (w1, w2):  # canonical FP4 zero signs, as B12xExperts loads them
        magnitude = packed & 0x77
        nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11
        packed.bitwise_and_(0x77 | (nonzero << 3))
    ones = torch.ones(E, dtype=torch.float32, device="cuda")
    weight_plan = fm.plan_weights(
        source=fm.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=fm.ActivationSpec(
            mode="a8",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=LIMIT,
            swiglu_alpha=None,
            swiglu_beta=None,
        ),
        geometry=fm.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=N),
    )
    prepared = fm.prepare_weights(
        plan=weight_plan,
        weights=fm.PackedWeights(
            w13=w1,
            w2=w2,
            w13_block_scales=w1s,
            w2_block_scales=w2s,
            w13_global_scales=ones,
            w2_global_scales=ones,
            input_scale=ones,
            intermediate_scale=ones,
            immutable_input_scales=True,
        ),
    )
    plan = fm.plan_execution(
        experts=prepared,
        capacity=fm.ExecutionCapacity(
            max_tokens=TOKENS, top_k=TOPK, warmup_token_counts=(TOKENS,), route_num_experts=0
        ),
        routing=fm.RoutingSpec(apply_router_weight_on_input=False),
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.zeros(int(spec.shape[0]), dtype=torch.uint8, device="cuda")

    def run(x, ids, w):
        out = torch.empty_like(x)
        binding = fm.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=prepared,
            topk_weights=w,
            topk_ids=ids,
            output=out,
            input_scales_static=True,
        )
        fm.run(binding=binding)
        return out

    def prepare_call(state):
        x = torch.zeros(TOKENS, H, dtype=torch.bfloat16, device="cuda")
        ids = (torch.arange(TOPK, device="cuda", dtype=torch.int32).expand(TOKENS, -1) % E).contiguous()
        w = torch.full((TOKENS, TOPK), 1.0 / TOPK, dtype=torch.float32, device="cuda")
        binding = state.bind(
            scratch=scratch,
            a=x,
            experts=prepared,
            topk_weights=w,
            topk_ids=ids,
            output=torch.empty_like(x),
            input_scales_static=True,
        )
        return PreparedCall(run=lambda: fm.run(binding=binding))

    prepare_default(plan.request(name="fused-decode-test", prepare_call=prepare_call))
    return micro, original, fused, run


def _routing(unique: int, seed: int, padding: int):
    g = torch.Generator().manual_seed(seed)
    pool = torch.randperm(E, generator=g)[:unique].tolist()
    rows, used = [], set()
    for _ in range(TOKENS):
        must = [e for e in pool if e not in used][:TOPK]
        rest = [pool[i] for i in torch.randperm(unique, generator=g).tolist() if pool[i] not in must]
        row = (must + rest)[:TOPK]
        used.update(row)
        rows.append(row)
    for t in range(TOKENS - padding, TOKENS):
        rows[t] = [-1] * TOPK  # padding tokens, as the router writes them
    w = torch.rand(TOKENS, TOPK, generator=g) + 0.1
    x = torch.randn(TOKENS, H, generator=g) * 0.5
    return (
        x.to(torch.bfloat16).cuda(),
        torch.tensor(rows, dtype=torch.int32).cuda(),
        (w / w.sum(-1, keepdim=True) * 1.5).cuda(),
    )


@pytest.mark.parametrize("unique,seed,padding", [(12, 1, 0), (24, 2, 0), (36, 3, 0), (44, 4, 0), (24, 5, 3)])
def test_fused_decode_moe_matches_b12x(moe, monkeypatch, unique, seed, padding):
    micro, original, fused, run = moe
    x, ids, w = _routing(unique, seed, padding)
    monkeypatch.setattr(micro, "launch_w4a8_compact_micro", original)
    expected = run(x, ids, w)
    monkeypatch.setattr(micro, "launch_w4a8_compact_micro", fused)
    actual = run(x, ids, w)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


def test_fused_decode_moe_replays_from_a_cuda_graph(moe, monkeypatch):
    micro, _, fused, run = moe
    monkeypatch.setattr(micro, "launch_w4a8_compact_micro", fused)
    x, ids, w = _routing(30, 6, 0)
    expected = run(x, ids, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(x, ids, w)
    for _ in range(10):  # the kernel's last CTA resets the control block for the next replay
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
