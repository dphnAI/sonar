# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 query and index-K cache must match a dequantized FP32 ranking reference.

The MiniMax M3 lightning-indexer scores only feed a top-k block ranking (the
attention itself reads the main KV cache), so an fp8 (e4m3) side cache trades
~2% relative score error for half the per-step index-K read bandwidth — the
dominant linear-in-context decode cost at long context. #45892 enabled fp8
index caches on the SM100 MSA path; this covers the Triton impl used
everywhere else (e.g. SM120, ROCm gfx942), for both the decode and prefill
scorers.
"""

import pytest
import torch

from sonar.platforms import current_platform

if current_platform.is_rocm():
    from sonar.models.minimax_m3.amd.ops.index_topk import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_index_decode,
        minimax_m3_index_score,
        minimax_m3_index_topk,
    )
    from sonar.platforms.rocm import on_gfx942

    FP8_SUPPORTED = on_gfx942()
else:
    from sonar.models.minimax_m3.common.ops.index_topk import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_index_decode,
        minimax_m3_index_score,
        minimax_m3_index_topk,
    )

    FP8_SUPPORTED = current_platform.is_cuda() and current_platform.supports_fp8()
FP8_DTYPE = current_platform.fp8_dtype()
PAGE = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
HEADS = 1
TOPK = 16
NUM_PAGES = 100
SEQ_LEN = 96 * PAGE + 37
PLANTED = (7, 40, 88)


def _make_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    keys = torch.randn(NUM_PAGES * PAGE, HEAD_DIM, dtype=torch.bfloat16, device=device)
    # RMS-normalize like the real (post index_k_norm) keys.
    keys = keys * keys.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt().to(torch.bfloat16)
    q = torch.randn(1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    for b in PLANTED:
        keys[b * PAGE + 5] = (q[0, 0] * 3).to(torch.bfloat16)
    return q, keys


def _check_selection(selected: set[int], query: torch.Tensor, keys: torch.Tensor) -> None:
    assert len(selected) == TOPK
    assert all(b in selected for b in PLANTED)
    token_scores = (keys[:SEQ_LEN].float() * query[0, 0].float()).sum(dim=-1)
    padding = (-SEQ_LEN) % PAGE
    token_scores = torch.nn.functional.pad(token_scores, (0, padding), value=-torch.inf)
    block_scores = token_scores.reshape(-1, PAGE).amax(dim=-1)
    assert all(0 <= b < block_scores.numel() for b in selected)
    expected = block_scores.topk(TOPK).values
    actual = block_scores[sorted(selected)].sort(descending=True).values
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-4)


@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 support required")
@torch.inference_mode()
def test_fp8_index_query_and_cache_decode_topk_matches_reference() -> None:
    dev = "cuda"
    q, keys = _make_inputs(dev)
    block_table = torch.arange(NUM_PAGES, dtype=torch.int32, device=dev).unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev)

    def select(cache: torch.Tensor) -> set[int]:
        query = q.to(cache.dtype)
        idx = minimax_m3_index_decode(
            query,
            cache.view(NUM_PAGES, PAGE, HEAD_DIM).contiguous(),
            block_table,
            seq_lens,
            SEQ_LEN,
            TOPK,
            0,
            0,
            HEADS,
            1,
            1,
        )
        return {int(b) for b in idx[0, 0].tolist() if b >= 0}

    for cache in (keys, keys.to(FP8_DTYPE)):
        _check_selection(select(cache), q.to(cache.dtype), cache)


@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 support required")
@torch.inference_mode()
def test_fp8_index_query_and_cache_prefill_topk_matches_reference() -> None:
    dev = "cuda"
    q, keys = _make_inputs(dev)
    block_table = torch.arange(NUM_PAGES, dtype=torch.int32, device=dev).unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev)
    # single decode-like query token at the end of the sequence
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    prefix_lens = torch.tensor([SEQ_LEN - 1], dtype=torch.int32, device=dev)

    def select(cache: torch.Tensor) -> set[int]:
        query = q.to(cache.dtype)
        score = minimax_m3_index_score(
            query,
            cache.view(NUM_PAGES, PAGE, HEAD_DIM).contiguous(),
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            1,
            SEQ_LEN,
            HEADS,
        )
        idx = minimax_m3_index_topk(score, cu_seqlens_q, prefix_lens, 1, TOPK, 0, 0)
        return {int(b) for b in idx[0, 0].tolist() if b >= 0}

    for cache in (keys, keys.to(FP8_DTYPE)):
        _check_selection(select(cache), q.to(cache.dtype), cache)


@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 support required")
@torch.inference_mode()
def test_bf16_query_against_fp8_cache_decode_topk_matches_reference() -> None:
    """A mixed pair must upcast rather than emit a mixed-dtype fp8 dot.

    Production allocates index_q with the index-K cache dtype, so the scorers
    normally see a matched pair and keep the native FP8 MMA. This pins the
    other branch: a bf16 query against an e4m3 cache still compiles and still
    ranks the planted blocks.
    """
    dev = "cuda"
    q, keys = _make_inputs(dev)
    block_table = torch.arange(NUM_PAGES, dtype=torch.int32, device=dev).unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev)

    def select(cache: torch.Tensor) -> set[int]:
        idx = minimax_m3_index_decode(
            q,  # bf16 query, deliberately not cast to the cache dtype
            cache.view(NUM_PAGES, PAGE, HEAD_DIM).contiguous(),
            block_table,
            seq_lens,
            SEQ_LEN,
            TOPK,
            0,
            0,
            HEADS,
            1,
            1,
        )
        return {int(b) for b in idx[0, 0].tolist() if b >= 0}

    for cache in (keys, keys.to(FP8_DTYPE)):
        _check_selection(select(cache), q, cache)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and FP8_SUPPORTED),
    reason="ROCm gfx942 Triton index-cache insert",
)
@torch.inference_mode()
def test_fp8_index_cache_insert_saturates() -> None:
    from sonar.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from sonar.models.minimax_m3.amd.ops.sparse_pa import (
        minimax_m3_insert_index_cache,
    )

    dev = "cuda"
    index_k = torch.tensor(
        [[1e4] * HEAD_DIM, [-1e4] * HEAD_DIM, [1.5] * HEAD_DIM],
        dtype=torch.bfloat16,
        device=dev,
    )
    cache = torch.zeros(1, PAGE, HEAD_DIM, dtype=FP8_DTYPE, device=dev)
    slots = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)
    minimax_m3_insert_index_cache(index_k, cache, slots)
    rows = cache[0, :3].float()
    fp8_max = get_fp8_min_max()[1]
    assert torch.equal(rows[0], torch.full_like(rows[0], fp8_max))
    assert torch.equal(rows[1], torch.full_like(rows[1], -fp8_max))
    assert torch.equal(rows[2], torch.full_like(rows[2], 1.5))
