# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41_sparse_decode_sm120 against b12x's pure-torch DeepSeek-V4.1 sparse MLA reference."""

import pytest
import torch

from aphrodite.platforms import current_platform

reference = pytest.importorskip("b12x.attention._shared.mla.compressed_reference")

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(120) or not hasattr(torch.ops._C, "dsv41_sparse_decode_sm120"),
    reason="requires the SM120 sparse decode kernel",
)

H, D, WINDOW, TOPK, SWA_PAGE = 16, 512, 128, 512, 128
SCALE = 512**-0.5


def _case(*, seed, ratio, positions_per_request, visible_cap=10**9, padding_rows=0, hole=False):
    """Encode random caches and build the metadata the b12x decode path consumes."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    indexed = ratio > 0
    idx_page = 256 // ratio if indexed else 1
    reqs = [r for r, positions in enumerate(positions_per_request) for _ in positions]
    positions = [p for ps in positions_per_request for p in ps]
    rows = len(positions) + padding_rows
    nreq = len(positions_per_request)
    max_pos = max(positions) + 1
    swa_pages = -(-max_pos // SWA_PAGE)
    comp = max(max_pos // ratio, 1) if indexed else 1
    idx_pages = -(-comp // idx_page)
    width = idx_pages + 2
    swa_phys = (torch.randperm(nreq * swa_pages, generator=gen, device="cuda") + 1).view(nreq, swa_pages)
    idx_phys = (torch.randperm(nreq * idx_pages, generator=gen, device="cuda") + 1).view(nreq, idx_pages)
    swa_cache = reference.pack_deepseek_v41_cache_reference(
        torch.randn((nreq * swa_pages + 1) * SWA_PAGE, D, generator=gen, device="cuda"),
        page_size=SWA_PAGE,
        cache_kind="swa",
    )
    idx_cache = reference.pack_deepseek_v41_cache_reference(
        torch.randn((nreq * idx_pages + 1) * idx_page, D, generator=gen, device="cuda"),
        page_size=idx_page,
        cache_kind="indexed",
    )
    q = torch.randn(rows, H, D, generator=gen, device="cuda").to(torch.bfloat16)
    sink = torch.randn(H, generator=gen, device="cuda")
    swa_slots = torch.full((rows, WINDOW), -1, dtype=torch.int32, device="cuda")
    swa_len = torch.zeros(rows, dtype=torch.int32, device="cuda")
    logical = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    idx_slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    top_len = torch.zeros(rows, dtype=torch.int32, device="cuda")
    table = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    for row, (req, pos) in enumerate(zip(reqs, positions)):
        start = max(pos + 1 - WINDOW, 0)
        length = pos + 1 - start
        tokens = torch.arange(start, start + length, device="cuda")
        swa_slots[row, :length] = (swa_phys[req][tokens // SWA_PAGE] * SWA_PAGE + tokens % SWA_PAGE).to(torch.int32)
        swa_len[row] = length
        if not indexed:
            continue
        visible = min(max((pos + 1) // ratio, 1), comp, visible_cap)
        count = min(visible, TOPK)
        chosen = torch.randperm(visible, generator=gen, device="cuda")[:count].sort().values
        logical[row, :count] = chosen.to(torch.int32)
        pages = idx_phys[req][chosen // idx_page]
        slots = pages * idx_page + chosen % idx_page
        table[row, :idx_pages] = idx_phys[req].to(torch.int32)
        if hole:
            # A missing page-table entry masks every key on that page.
            missing = int(chosen[0]) // idx_page
            table[row, missing] = -1
            slots = slots[chosen // idx_page != missing]
        idx_slots[row, : slots.numel()] = slots.to(torch.int32)
        top_len[row] = count
    return dict(
        q=q,
        sink=sink,
        swa_cache=swa_cache,
        idx_cache=idx_cache,
        idx_page=idx_page,
        indexed=indexed,
        swa_slots=swa_slots,
        swa_len=swa_len,
        logical=logical,
        idx_slots=idx_slots,
        top_len=top_len,
        table=table,
    )


def _run(c):
    rows = c["q"].shape[0]
    out = torch.empty_like(c["q"])
    partial = torch.empty(rows * 8 * H * D, dtype=torch.float32, device="cuda")
    stats = torch.empty(rows * 8 * H * 2, dtype=torch.float32, device="cuda")
    indexed = c["indexed"]
    torch.ops._C.dsv41_sparse_decode_sm120(
        c["q"],
        out,
        c["swa_cache"],
        SWA_PAGE,
        c["swa_slots"],
        c["swa_len"],
        c["idx_cache"] if indexed else None,
        c["idx_page"],
        c["logical"] if indexed else None,
        c["top_len"] if indexed else None,
        c["table"] if indexed else None,
        c["sink"],
        SCALE,
        partial,
        stats,
    )
    return out


def _reference(c):
    # Masked indexed keys are dropped from the reference's prefix.
    top = (c["idx_slots"] >= 0).sum(dim=1).to(torch.int32)
    kwargs = {}
    if c["indexed"]:
        kwargs = dict(
            extra_k_cache=c["idx_cache"],
            extra_indices=c["idx_slots"],
            extra_topk_lengths=top,
            extra_page_size=c["idx_page"],
        )
    return reference.compressed_sparse_mla_reference(
        c["q"].float(),
        c["swa_cache"],
        c["swa_slots"],
        c["swa_len"],
        sm_scale=SCALE,
        attn_sink=c["sink"],
        swa_page_size=SWA_PAGE,
        cache_format="deepseek_v41",
        **kwargs,
    ).float()


@pytest.mark.parametrize(
    "spec",
    [
        dict(seed=0, ratio=4, positions_per_request=[list(range(2674, 2682))]),
        dict(seed=1, ratio=4, positions_per_request=[list(range(40, 48))]),
        dict(seed=2, ratio=4, positions_per_request=[list(range(5000, 5004)), list(range(300, 304))], visible_cap=200),
        dict(seed=3, ratio=128, positions_per_request=[list(range(9000, 9008))]),
        dict(seed=4, ratio=128, positions_per_request=[list(range(100, 104)), list(range(70000, 70004))]),
        dict(seed=5, ratio=0, positions_per_request=[list(range(60, 68))]),
        dict(seed=6, ratio=4, positions_per_request=[list(range(4096, 4102))], padding_rows=2),
        dict(seed=7, ratio=4, positions_per_request=[list(range(3000, 3008))], hole=True),
    ],
)
def test_matches_reference_and_is_deterministic(spec):
    c = _case(**spec)
    actual = _run(c)
    expected = _reference(c)
    rel = ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    assert rel < 3e-3, f"relative L2 error {rel:.2e}"
    padding = spec.get("padding_rows", 0)
    if padding:
        # No keys and only the sink: the output is exactly zero.
        assert torch.count_nonzero(actual[-padding:]) == 0
    assert torch.equal(actual, _run(c))
