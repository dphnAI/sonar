# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the sm89 (Ada / RTX 4090) DeepSeek sparse attention kernels.

Covers the four custom ops
  - sm89_paged_mqa_logits_metadata, the (request, page) partition table
  - sm89_fp8_paged_mqa_logits, paged decode indexer logits
  - sm89_fp8_mqa_logits, ragged prefill indexer logits
  - sm89_sparse_mla_fwd, sparse MLA forward from an fp8_ds_mla pool

The synthetic pool builders are byte-exact mirrors of the in-tree cache writers
in csrc/libtorch_stable/cache_kernels.cu. The indexer K cache stores each
64-token page as [64x128 fp8 keys, token-major][64 fp32 per-token scales]
(8448 B/page, scale = max(amax, 1e-4) / 448, optionally rounded up to a power
of two), and the fp8_ds_mla cache stores each token as [512 fp8 nope][4 fp32
per-128-tile scales][64 bf16 rope] (656 B/token).
"""

import pytest
import torch
import torch.nn.functional as F

from aphrodite import _custom_ops as ops
from aphrodite.platforms import current_platform
from aphrodite.utils.math_utils import cdiv

PAGE = 64
HEAD_DIM = 128
NUM_HEADS = 64
PAGE_BYTES = PAGE * (HEAD_DIM + 4)
FP8_MAX = 448.0

DS_NOPE = 512
DS_ROPE = 64
DS_TILE = 128
DS_BYTES = DS_NOPE + (DS_NOPE // DS_TILE) * 4 + DS_ROPE * 2  # 656

NEG_INF = float("-inf")


def _sm89_dsa_ready() -> bool:
    return (
        current_platform.is_cuda()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 9)
        and ops.supports_sm89_dsa()
    )


pytestmark = pytest.mark.skipif(not _sm89_dsa_ready(), reason="requires an sm89 build with the DSA kernels")


# --------------------------------------------------------------------------- input builders
def _quant_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _quant_indexer_tokens(k: torch.Tensor, ue8m0: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token fp8 quantization matching indexer_k_quant_and_cache_kernel.

    Returns (k_fp8 [N, D], k_deq [N, D] fp32, scale [N] fp32); k_deq is the
    exact value a correct kernel reconstructs (fp8 value * scale).
    """
    amax = k.abs().amax(dim=-1).clamp_min(1e-4)
    scale = amax / FP8_MAX
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    k_fp8 = _quant_e4m3(k, scale[:, None])
    k_deq = k_fp8.float() * scale[:, None]
    return k_fp8, k_deq, scale


def _make_indexer_q(num_rows: int, num_heads: int = NUM_HEADS) -> tuple[torch.Tensor, torch.Tensor]:
    """q fp8 [M, H, 128] plus weights fp32 [M, H] with q_scale folded in
    (as the indexer does), keeping logits O(1)."""
    q = torch.randn(num_rows, num_heads, HEAD_DIM, device="cuda")
    q_amax = q.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    q_scale = q_amax / FP8_MAX
    q_fp8 = _quant_e4m3(q, q_scale)
    weights = torch.rand(num_rows, num_heads, device="cuda") * (q_scale.squeeze(-1) * HEAD_DIM**-0.5 * num_heads**-0.5)
    return q_fp8, weights


def _ke_2d(seq_bases: list[int], next_n: int) -> torch.Tensor:
    """Per-token key windows [B, next_n] i32. Token t of request b attends to
    seq_bases[b] - (next_n - 1) + t keys (speculative decode layout)."""
    base = torch.tensor(seq_bases, dtype=torch.int32)
    t = torch.arange(next_n, dtype=torch.int32)
    return (base[:, None] - (next_n - 1) + t[None, :]).clamp_min_(0).cuda()


def _build_k1_case(
    seq_bases: list[int],
    next_n: int,
    ue8m0: bool = True,
    max_pages: int | None = None,
    num_heads: int = NUM_HEADS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (q_fp8 [B, next_n, H, 128], pool u8 [P, 8448], weights,
    seq_lens [B, next_n] i32, block_table [B, max_pages] i32).

    Each request's pages are scattered to non-contiguous physical pool pages.
    Padding tokens past ke within the last page hold real (quantized) data so
    the kernel's tail masking is actually exercised.
    """
    B = len(seq_bases)
    ke = _ke_2d(seq_bases, next_n)
    pages_per_req = [cdiv(int(ke[b].max()), PAGE) for b in range(B)]
    total_pages = sum(pages_per_req)
    max_pages = max_pages or max(pages_per_req + [1])
    pool_pages = total_pages + 4
    perm = torch.randperm(pool_pages, device="cuda")[: max(total_pages, 1)]

    pool = torch.zeros(pool_pages, PAGE_BYTES, dtype=torch.uint8, device="cuda")
    block_table = torch.zeros(B, max_pages, dtype=torch.int32, device="cuda")
    next_free = 0
    for b in range(B):
        n_pages = pages_per_req[b]
        if n_pages == 0:
            continue
        phys = perm[next_free : next_free + n_pages]
        next_free += n_pages
        block_table[b, :n_pages] = phys.to(torch.int32)
        k = torch.randn(n_pages * PAGE, HEAD_DIM, device="cuda")
        k_fp8, _, scale = _quant_indexer_tokens(k, ue8m0)
        pool[phys, : PAGE * HEAD_DIM] = k_fp8.view(torch.uint8).reshape(n_pages, PAGE * HEAD_DIM)
        pool[phys, PAGE * HEAD_DIM :] = scale.reshape(n_pages, PAGE).view(torch.uint8).reshape(n_pages, PAGE * 4)

    q_fp8, weights = _make_indexer_q(B * next_n, num_heads)
    return q_fp8.reshape(B, next_n, num_heads, HEAD_DIM), pool, weights, ke, block_table


def _make_ds_mla_pool(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pool u8 [S, 656], kv_deq [S, 576] fp32); kv_deq[:, :512] is the
    dequantized nope (fp8 value * per-tile scale), [:, 512:] the rope (bf16 exact)."""
    kv = torch.randn(num_tokens, DS_NOPE + DS_ROPE, device="cuda")
    nope = kv[:, :DS_NOPE].reshape(num_tokens, DS_NOPE // DS_TILE, DS_TILE)
    amax = nope.abs().amax(dim=-1)
    scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    nope_fp8 = _quant_e4m3(nope, scale[..., None])
    nope_deq = nope_fp8.float() * scale[..., None]
    rope_bf16 = kv[:, DS_NOPE:].to(torch.bfloat16)

    pool = torch.zeros(num_tokens, DS_BYTES, dtype=torch.uint8, device="cuda")
    pool[:, :DS_NOPE] = nope_fp8.reshape(num_tokens, DS_NOPE).view(torch.uint8)
    pool[:, DS_NOPE : DS_NOPE + 16] = scale.view(torch.uint8).reshape(num_tokens, 16)
    pool[:, DS_NOPE + 16 :] = rope_bf16.view(torch.uint8).reshape(num_tokens, DS_ROPE * 2)

    kv_deq = torch.cat([nope_deq.reshape(num_tokens, DS_NOPE), rope_bf16.float()], dim=-1)
    return pool, kv_deq


def _make_sparse_indices(num_tokens: int, topk: int, num_slots: int) -> torch.Tensor:
    """[T, topk] i32 physical slots, tail -1 padded, unique per row.

    Row 0 is all -1 when T > 1 (fully padded query); one row is forced full and
    one to a non-multiple-of-32 valid count.
    """
    indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device="cuda")
    max_valid = min(topk, num_slots)
    valid = torch.randint(1, max_valid + 1, (num_tokens,)).tolist()
    valid[-1] = max(1, max_valid - 3)
    if num_tokens > 1:
        valid[0] = 0
        valid[1] = max_valid
    for t in range(num_tokens):
        if valid[t] == 0:
            continue
        slots = torch.randperm(num_slots, device="cuda")[: valid[t]]
        indices[t, : valid[t]] = slots.to(torch.int32)
    return indices


# --------------------------------------------------------------------------- reference oracles
def _mqa_logits_ref(
    q_fp8: torch.Tensor,
    k_deq: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    fill: float = 0.0,
    chunk: int = 64,
) -> torch.Tensor:
    """logits[m, n] = sum_h w[m,h] * relu(q[m,h] . k_deq[n]) for n in [ks_m, ke_m), else fill.

    Accumulates in float64 so the oracle is strictly more precise than the kernel.
    """
    M = q_fp8.shape[0]
    N = k_deq.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=q_fp8.device)
    kd = k_deq.double()
    for m0 in range(0, M, chunk):
        m1 = min(m0 + chunk, M)
        s = torch.einsum("mhd,nd->mhn", q_fp8[m0:m1].double(), kd)
        s = s.clamp(min=0.0) * weights[m0:m1, :, None].double()
        out[m0:m1] = s.sum(dim=1).float()
    n = torch.arange(N, device=out.device)[None, :]
    valid = (n >= ks[:, None]) & (n < ke[:, None])
    return torch.where(valid, out, torch.full_like(out, fill))


def _paged_mqa_logits_ref(
    q_fp8: torch.Tensor,
    pool: torch.Tensor,
    weights: torch.Tensor,
    ke_2d: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    fill: float = 0.0,
) -> torch.Tensor:
    """Assemble each request's logical K sequence from the paged pool via the
    block table, then apply the dense fp64 reference."""
    B, next_n, _, _ = q_fp8.shape
    keys = pool[:, : PAGE * HEAD_DIM].contiguous().view(torch.float8_e4m3fn)
    keys = keys.reshape(-1, PAGE, HEAD_DIM).float()
    scales = pool[:, PAGE * HEAD_DIM :].contiguous().view(torch.float32).reshape(-1, PAGE)

    out = torch.full((B * next_n, max_model_len), fill, dtype=torch.float32, device=q_fp8.device)
    for b in range(B):
        ke_req = int(ke_2d[b].max())
        if ke_req == 0:
            continue
        pages = block_table[b, : cdiv(ke_req, PAGE)].long()
        k_deq = (keys[pages] * scales[pages][..., None]).reshape(-1, HEAD_DIM)[:ke_req]
        for t in range(next_n):
            m = b * next_n + t
            ke = int(ke_2d[b, t])
            if ke == 0:
                continue
            s = torch.einsum("hd,nd->hn", q_fp8[b, t].double(), k_deq[:ke].double())
            out[m, :ke] = (s.clamp(min=0.0) * weights[m, :, None].double()).sum(dim=0).float()
    return out


def _sparse_mla_ref(
    q: torch.Tensor,
    kv_deq: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """out[t] = softmax(q_t . kv[idx_t]^T * sm_scale) @ kv[idx_t][:, :512]."""
    T, h, _ = q.shape
    out = torch.zeros(T, h, DS_NOPE, dtype=torch.float32, device=q.device)
    lse = torch.full((T, h), NEG_INF, dtype=torch.float32, device=q.device)
    for t in range(T):
        valid = indices[t] >= 0
        if not valid.any():
            continue
        rows = kv_deq[indices[t, valid].long()]
        s = q[t].float() @ rows.T * sm_scale
        out[t] = torch.softmax(s, dim=-1) @ rows[:, :DS_NOPE]
        lse[t] = torch.logsumexp(s, dim=-1)
    return out, lse


def _topk_set_agreement(
    logits_kernel: torch.Tensor,
    logits_oracle: torch.Tensor,
    ke: torch.Tensor,
    k: int = 2048,
    tie_tol: float = 1e-4,
) -> float:
    """Fraction of rows whose top-k index SETS agree, tie-aware.

    Distinct summation orders legitimately flip membership between values
    within fp32 noise of the k-th threshold. A flip is benign iff the oracle
    logit of every flipped index lies within tie_tol (relative) of the oracle's
    k-th value; any other disagreement fails the row.
    """
    M = logits_kernel.shape[0]
    agree = 0
    for m in range(M):
        L = int(ke[m])
        kk = min(k, L)
        if kk == 0:
            agree += 1
            continue
        top_kernel = set(torch.topk(logits_kernel[m, :L], kk).indices.tolist())
        vals, idx = torch.topk(logits_oracle[m, :L], kk)
        top_oracle = set(idx.tolist())
        if top_kernel == top_oracle:
            agree += 1
            continue
        thresh = vals[-1].item()
        tol = tie_tol * max(abs(thresh), 1.0)
        flipped = top_kernel.symmetric_difference(top_oracle)
        if all(abs(logits_oracle[m, i].item() - thresh) <= tol for i in flipped):
            agree += 1
    return agree / M


def _max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max |a-b| / max(|b|, 1), relative for O(1)+ logits and absolute below 1."""
    if a.numel() == 0:
        return 0.0
    return ((a - b).abs() / b.abs().clamp_min(1.0)).max().item()


def _ref_partition_table(ke_2d: torch.Tensor, num_partitions: int) -> torch.Tensor:
    """Pure-Python re-derivation of the partition-metadata kernel's output."""
    rows_ke = ke_2d.cpu().tolist()
    pages = [cdiv(max(row), PAGE) for row in rows_ke]
    prefix = [0]
    for p in pages:
        prefix.append(prefix[-1] + p)
    total, B = prefix[-1], len(pages)
    sched = []
    for p in range(num_partitions + 1):
        if total == 0:
            sched.append((-1, 0))
            continue
        target = p * total // num_partitions
        if target >= total:
            sched.append((B - 1, total - prefix[B - 1]))
            continue
        lo, hi = 0, B - 1
        while lo < hi:  # first request whose page prefix passes target
            mid = (lo + hi) // 2
            if prefix[mid + 1] > target:
                hi = mid
            else:
                lo = mid + 1
        sched.append((lo, target - prefix[lo]))
    return torch.tensor(sched, dtype=torch.int32)


def _run_paged_logits(
    q_fp8: torch.Tensor,
    pool: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
    num_partitions: int = 256,
) -> torch.Tensor:
    sched = torch.empty(num_partitions + 1, 2, dtype=torch.int32, device="cuda")
    ops.sm89_paged_mqa_logits_metadata(seq_lens, sched, seq_lens.shape[1])
    logits = torch.zeros(seq_lens.numel(), max_model_len, dtype=torch.float32, device="cuda")
    ops.sm89_fp8_paged_mqa_logits(
        q_fp8.view(torch.uint8), pool, weights, seq_lens, block_table, sched, logits, clean_logits
    )
    return logits


def _check_paged_logits(
    logits: torch.Tensor,
    ref: torch.Tensor,
    ke_2d: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> None:
    B, next_n = ke_2d.shape
    for m in range(B * next_n):
        b, t = divmod(m, next_n)
        ke = int(ke_2d[b, t])
        n_pages = cdiv(int(ke_2d[b].max()), PAGE)
        assert _max_rel_err(logits[m, :ke], ref[m, :ke]) <= 2e-3, f"row {m}"
        if clean_logits and n_pages > 0:
            # only the request's last page gets the -inf tail; everything else
            # past ke must be untouched (still the zero fill)
            lp0, lp1 = (n_pages - 1) * PAGE, n_pages * PAGE
            assert torch.isneginf(logits[m, max(ke, lp0) : lp1]).all(), f"row {m}"
            assert (logits[m, ke:lp0] == 0).all(), f"row {m}"
            assert (logits[m, lp1:] == 0).all(), f"row {m}"
        else:
            assert (logits[m, ke:] == 0).all(), f"row {m}"
    assert _topk_set_agreement(logits, ref, ke_2d.reshape(-1), k=2048) == 1.0, "top-2048 set mismatch"


def _graph_capture(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


# --------------------------------------------------------------------------- partition metadata
_MIXED_16 = [1, 17, 63, 64, 65, 127, 200, 333, 512, 700, 1024, 1500, 2047, 2048, 2049, 4000]

_META_CASES = [
    pytest.param([0], id="empty"),
    pytest.param([2048], id="single"),
    pytest.param([1, 63, 64, 65, 0, 4000], id="page-edges"),
    pytest.param(_MIXED_16, id="b16-mixed"),
]


@pytest.mark.parametrize("seq_bases", _META_CASES)
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("num_partitions", [1, 256])
def test_sm89_paged_mqa_logits_metadata(seq_bases, next_n, num_partitions):
    seq_lens = _ke_2d(seq_bases, next_n)
    sched = torch.empty(num_partitions + 1, 2, dtype=torch.int32, device="cuda")
    ops.sm89_paged_mqa_logits_metadata(seq_lens, sched, next_n)
    assert torch.equal(sched.cpu(), _ref_partition_table(seq_lens, num_partitions))


# --------------------------------------------------------------------------- paged decode logits
_K1_CASES = [
    # N in {2047, 2048, 2049} spans the SoA >2048 regression window around the
    # page holding key 2048
    pytest.param([2047], True, id="n2047"),
    pytest.param([2048], True, id="n2048"),
    pytest.param([2049], True, id="n2049"),
    pytest.param([2048], False, id="n2048-fp32-scales"),
    pytest.param([1, 63, 64, 65, 0], True, id="page-edges"),
    pytest.param(_MIXED_16, True, id="b16-mixed"),
]


@pytest.mark.parametrize(("seq_bases", "ue8m0"), _K1_CASES)
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("clean_logits", [False, True])
@pytest.mark.parametrize("num_heads", [32, 64])
def test_sm89_fp8_paged_mqa_logits(seq_bases, ue8m0, next_n, clean_logits, num_heads):
    torch.manual_seed(20260705)
    max_model_len = cdiv(max(seq_bases), PAGE) * PAGE + PAGE
    q_fp8, pool, weights, ke_2d, block_table = _build_k1_case(seq_bases, next_n, ue8m0, num_heads=num_heads)
    logits = _run_paged_logits(q_fp8, pool, weights, ke_2d, block_table, max_model_len, clean_logits)
    ref = _paged_mqa_logits_ref(q_fp8, pool, weights, ke_2d, block_table, max_model_len)
    _check_paged_logits(logits, ref, ke_2d, max_model_len, clean_logits)


def test_sm89_fp8_paged_mqa_logits_long_context():
    torch.manual_seed(31337)
    seq_bases = [131072]
    max_model_len = 131072 + PAGE
    q_fp8, pool, weights, ke_2d, block_table = _build_k1_case(seq_bases, next_n=1)
    logits = _run_paged_logits(q_fp8, pool, weights, ke_2d, block_table, max_model_len, clean_logits=False)
    ref = _paged_mqa_logits_ref(q_fp8, pool, weights, ke_2d, block_table, max_model_len)
    _check_paged_logits(logits, ref, ke_2d, max_model_len, clean_logits=False)


def test_sm89_paged_mqa_logits_cuda_graph():
    torch.manual_seed(42)
    B, next_n, max_model_len = 2, 2, 2048
    capture_bases, grown_bases = [513, 65], [1500, 900]
    q_fp8, pool, weights, _, block_table = _build_k1_case(grown_bases, next_n, max_pages=cdiv(max(grown_bases), PAGE))
    seq_lens = _ke_2d(capture_bases, next_n)
    sched = torch.empty(257, 2, dtype=torch.int32, device="cuda")
    logits = torch.zeros(B * next_n, max_model_len, dtype=torch.float32, device="cuda")
    q_u8 = q_fp8.view(torch.uint8)

    def run():
        ops.sm89_paged_mqa_logits_metadata(seq_lens, sched, next_n)
        ops.sm89_fp8_paged_mqa_logits(q_u8, pool, weights, seq_lens, block_table, sched, logits, False)

    def eager() -> torch.Tensor:
        sched_e = torch.empty_like(sched)
        logits_e = torch.zeros_like(logits)
        ops.sm89_paged_mqa_logits_metadata(seq_lens, sched_e, next_n)
        ops.sm89_fp8_paged_mqa_logits(q_u8, pool, weights, seq_lens, block_table, sched_e, logits_e, False)
        torch.cuda.synchronize()
        return logits_e

    graph = _graph_capture(run)

    def replay() -> torch.Tensor:
        logits.zero_()
        graph.replay()
        torch.cuda.synchronize()
        return logits

    assert torch.equal(replay(), eager())

    # mutate q/weights in place; the captured pointers must see the new data
    torch.manual_seed(4242)
    q_new, w_new = _make_indexer_q(B * next_n)
    q_fp8.copy_(q_new.reshape(B, next_n, NUM_HEADS, HEAD_DIM))
    weights.copy_(w_new)
    assert torch.equal(replay(), eager())

    # with grown seq_lens the same graph must cover more pages (O(context) proof)
    seq_lens.copy_(_ke_2d(grown_bases, next_n))
    assert torch.equal(replay(), eager())


# --------------------------------------------------------------------------- ragged prefill logits
@pytest.mark.parametrize("M", [1, 64, 512])
@pytest.mark.parametrize("N", [1, 127, 128, 2047, 2048, 2049, 4096])
@pytest.mark.parametrize("window", ["full", "causal", "empty", "partial"])
@pytest.mark.parametrize("num_heads", [32, 64])
def test_sm89_fp8_mqa_logits(M, N, window, num_heads):
    torch.manual_seed(M * 100003 + N)
    q_fp8, weights = _make_indexer_q(M, num_heads)
    k = torch.randn(N, HEAD_DIM, device="cuda")
    # ue8m0=False gives arbitrary (non-power-of-two) fp32 scales
    k_fp8, k_deq, k_scales = _quant_indexer_tokens(k, ue8m0=False)

    if window == "full":
        ks = torch.zeros(M, dtype=torch.int32, device="cuda")
        ke = torch.full((M,), N, dtype=torch.int32, device="cuda")
    elif window == "causal":
        ks = torch.zeros(M, dtype=torch.int32, device="cuda")
        ke = (torch.arange(M, dtype=torch.int32, device="cuda") + (N - M) + 1).clamp_(0, N)
    elif window == "empty":
        ks = torch.full((M,), N // 2, dtype=torch.int32, device="cuda")
        ke = ks.clone()
    else:
        ks = torch.randint(0, N + 1, (M,), dtype=torch.int32, device="cuda")
        span = torch.randint(0, N + 1, (M,), dtype=torch.int32, device="cuda")
        ke = (ks + span).clamp_max_(N)

    logits = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    ops.sm89_fp8_mqa_logits(q_fp8.view(torch.uint8), k_fp8.view(torch.uint8), k_scales, weights, ks, ke, logits)

    ref = _mqa_logits_ref(q_fp8, k_deq, weights, ks, ke, fill=0.0)
    n = torch.arange(N, device="cuda")[None, :]
    in_window = (n >= ks[:, None]) & (n < ke[:, None])
    assert (logits[~in_window] == 0).all(), "kernel wrote outside its [ks, ke) window"
    if in_window.any():
        rel = (logits - ref).abs() / ref.abs().clamp_min(1.0)
        assert rel[in_window].max().item() <= 2e-3
    # tie_tol=5e-4 because the fp32-acc kernel's summation-order noise vs the
    # fp64 oracle is ~3e-4 at the top-k boundary; stays tie-aware without
    # flagging benign flips (still 4x inside the 2e-3 value bound)
    assert _topk_set_agreement(logits, ref, ke, k=2048, tie_tol=5e-4) == 1.0


# --------------------------------------------------------------------------- sparse MLA
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_sm89_sparse_mla_fwd(num_heads, topk, num_tokens):
    torch.manual_seed(num_heads * 10007 + topk + num_tokens)
    sm_scale = 576**-0.5
    num_slots = topk + 512
    pool, kv_deq = _make_ds_mla_pool(num_slots)
    q = torch.randn(num_tokens, num_heads, 576, device="cuda", dtype=torch.bfloat16)
    indices = _make_sparse_indices(num_tokens, topk, num_slots)

    out = torch.empty(num_tokens, num_heads, DS_NOPE, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(num_tokens, num_heads, device="cuda", dtype=torch.float32)
    ops.sm89_sparse_mla_fwd(q, pool, indices, out, lse, sm_scale)

    out_f = out.float()
    assert not torch.isnan(out_f).any() and not torch.isnan(lse).any()

    ref_out, ref_lse = _sparse_mla_ref(q, kv_deq, indices, sm_scale)
    valid = (indices >= 0).any(dim=-1)
    if (~valid).any():
        assert (out_f[~valid] == 0).all(), "all-(-1) rows must produce exact-0 output"
        assert torch.isneginf(lse[~valid]).all(), "all-(-1) rows must produce -inf LSE"
    cos = F.cosine_similarity(out_f[valid].reshape(-1, DS_NOPE), ref_out[valid].reshape(-1, DS_NOPE), dim=-1)
    assert cos.min().item() > 0.999, f"min cosine {cos.min().item()}"
    assert (out_f[valid] - ref_out[valid]).abs().max().item() < 0.1
    assert (lse[valid] - ref_lse[valid]).abs().max().item() < 3e-2


@pytest.mark.parametrize("num_heads", [16, 32])
def test_sm89_sparse_mla_fwd_topk_lens(num_heads):
    """topk_lens on front-compacted rows must reproduce the full loop bitwise.

    _make_sparse_indices already packs each row's valid slots to the front, so
    the row valid counts are exact topk_lens. The early exit only drops all-(-1)
    key tiles, which are exact no-ops in the full loop, so out/lse must match
    the topk_lens=None run byte for byte.
    """
    torch.manual_seed(20260706 + num_heads)
    num_tokens, topk, num_slots = 16, 2048, 2560
    sm_scale = 576**-0.5
    pool, _ = _make_ds_mla_pool(num_slots)
    q = torch.randn(num_tokens, num_heads, 576, device="cuda", dtype=torch.bfloat16)
    indices = _make_sparse_indices(num_tokens, topk, num_slots)
    # The helper already forces len == 0 (row 0) and len == topk (row 1); add a
    # row shorter than the cp.async prologue depth (len <= BI < STAGES * BI)
    # and a non-multiple-of-BI row (partial last tile).
    for t, n in ((2, 5), (3, 67)):
        indices[t] = -1
        indices[t, :n] = torch.randperm(num_slots, device="cuda")[:n].to(torch.int32)
    topk_lens = (indices >= 0).sum(dim=-1, dtype=torch.int32)

    def run(lens: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty(num_tokens, num_heads, DS_NOPE, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(num_tokens, num_heads, device="cuda", dtype=torch.float32)
        ops.sm89_sparse_mla_fwd(q, pool, indices, out, lse, sm_scale, lens)
        torch.cuda.synchronize()
        return out, lse

    out_full, lse_full = run(None)
    out_skip, lse_skip = run(topk_lens)
    # compare raw bit patterns, bitwise rather than just numeric equality
    assert torch.equal(out_skip.view(torch.int16), out_full.view(torch.int16))
    assert torch.equal(lse_skip.view(torch.int32), lse_full.view(torch.int32))


def test_sm89_sparse_mla_topk_lens_cuda_graph():
    """The lens-gated kernel must be capturable, with lens read at replay time."""
    torch.manual_seed(2077)
    num_tokens, num_heads, topk, num_slots = 8, 16, 2048, 2560
    sm_scale = 576**-0.5
    pool, _ = _make_ds_mla_pool(num_slots)
    q = torch.randn(num_tokens, num_heads, 576, device="cuda", dtype=torch.bfloat16)
    indices = _make_sparse_indices(num_tokens, topk, num_slots)
    topk_lens = (indices >= 0).sum(dim=-1, dtype=torch.int32)
    out = torch.zeros(num_tokens, num_heads, DS_NOPE, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(num_tokens, num_heads, device="cuda", dtype=torch.float32)

    graph = _graph_capture(lambda: ops.sm89_sparse_mla_fwd(q, pool, indices, out, lse, sm_scale, topk_lens))

    def replay_vs_full() -> None:
        out.zero_()
        lse.zero_()
        graph.replay()
        out_e = torch.zeros_like(out)
        lse_e = torch.zeros_like(lse)
        ops.sm89_sparse_mla_fwd(q, pool, indices, out_e, lse_e, sm_scale, None)
        torch.cuda.synchronize()
        assert torch.equal(out, out_e) and torch.equal(lse, lse_e)

    replay_vs_full()

    # mutate indices + lens in place; the captured pointers must see new data
    torch.manual_seed(2078)
    indices.copy_(_make_sparse_indices(num_tokens, topk, num_slots))
    topk_lens.copy_((indices >= 0).sum(dim=-1, dtype=torch.int32))
    q.copy_(torch.randn_like(q))
    replay_vs_full()


def test_sm89_sparse_mla_cuda_graph():
    torch.manual_seed(7)
    num_tokens, num_heads, topk, num_slots = 8, 16, 512, 1024
    sm_scale = 576**-0.5
    pool, _ = _make_ds_mla_pool(num_slots)
    q = torch.randn(num_tokens, num_heads, 576, device="cuda", dtype=torch.bfloat16)
    indices = _make_sparse_indices(num_tokens, topk, num_slots)
    out = torch.zeros(num_tokens, num_heads, DS_NOPE, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(num_tokens, num_heads, device="cuda", dtype=torch.float32)

    def run():
        ops.sm89_sparse_mla_fwd(q, pool, indices, out, lse, sm_scale)

    def eager() -> tuple[torch.Tensor, torch.Tensor]:
        out_e = torch.zeros_like(out)
        lse_e = torch.zeros_like(lse)
        ops.sm89_sparse_mla_fwd(q, pool, indices, out_e, lse_e, sm_scale)
        torch.cuda.synchronize()
        return out_e, lse_e

    graph = _graph_capture(run)

    def replay() -> None:
        out.zero_()
        lse.zero_()
        graph.replay()
        torch.cuda.synchronize()

    replay()
    out_e, lse_e = eager()
    assert torch.equal(out, out_e) and torch.equal(lse, lse_e)

    # mutate every input in place (new pool bytes, q, and index pattern)
    torch.manual_seed(77)
    pool_new, _ = _make_ds_mla_pool(num_slots)
    pool.copy_(pool_new)
    q.copy_(torch.randn_like(q))
    indices.copy_(_make_sparse_indices(num_tokens, topk, num_slots))
    replay()
    out_e, lse_e = eager()
    assert torch.equal(out, out_e) and torch.equal(lse, lse_e)


# --------------------------------------------------------------------------- determinism
def test_sm89_dsa_determinism():
    torch.manual_seed(123)

    # partition metadata + paged decode logits
    next_n, max_model_len = 2, 1088
    q_fp8, pool, weights, ke_2d, block_table = _build_k1_case([333, 1024, 63], next_n)
    sched_a = torch.empty(257, 2, dtype=torch.int32, device="cuda")
    sched_b = torch.empty_like(sched_a)
    ops.sm89_paged_mqa_logits_metadata(ke_2d, sched_a, next_n)
    ops.sm89_paged_mqa_logits_metadata(ke_2d, sched_b, next_n)
    assert torch.equal(sched_a, sched_b)
    runs = []
    for sched in (sched_a, sched_b):
        logits = torch.zeros(ke_2d.numel(), max_model_len, dtype=torch.float32, device="cuda")
        ops.sm89_fp8_paged_mqa_logits(q_fp8.view(torch.uint8), pool, weights, ke_2d, block_table, sched, logits, True)
        runs.append(logits)
    assert torch.equal(runs[0], runs[1])

    # ragged prefill logits
    M, N = 64, 2049
    q2_fp8, w2 = _make_indexer_q(M)
    k = torch.randn(N, HEAD_DIM, device="cuda")
    k_fp8, _, k_scales = _quant_indexer_tokens(k, ue8m0=False)
    ks = torch.randint(0, N + 1, (M,), dtype=torch.int32, device="cuda")
    ke = (ks + torch.randint(0, N + 1, (M,), dtype=torch.int32, device="cuda")).clamp_max_(N)
    runs = []
    for _ in range(2):
        logits = torch.zeros(M, N, dtype=torch.float32, device="cuda")
        ops.sm89_fp8_mqa_logits(q2_fp8.view(torch.uint8), k_fp8.view(torch.uint8), k_scales, w2, ks, ke, logits)
        runs.append(logits)
    assert torch.equal(runs[0], runs[1])

    # sparse MLA
    num_tokens, num_heads, topk, num_slots = 8, 16, 2048, 2560
    pool3, _ = _make_ds_mla_pool(num_slots)
    q3 = torch.randn(num_tokens, num_heads, 576, device="cuda", dtype=torch.bfloat16)
    indices = _make_sparse_indices(num_tokens, topk, num_slots)
    runs = []
    for _ in range(2):
        out = torch.empty(num_tokens, num_heads, DS_NOPE, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(num_tokens, num_heads, device="cuda", dtype=torch.float32)
        ops.sm89_sparse_mla_fwd(q3, pool3, indices, out, lse, 576**-0.5)
        runs.append((out, lse))
    assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])
