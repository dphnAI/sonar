# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 FP8 sparse-attention indexer score kernels."""

from functools import cache
from pathlib import Path

import torch

from aphrodite import envs
from aphrodite.platforms import current_platform
from aphrodite.triton_utils import tl, triton


@cache
def use_sm120_dsa_indexer(page_size: int | None = None) -> bool:
    """Whether SM120 needs the Triton fallback for this indexer operation."""
    if not (current_platform.is_cuda() and current_platform.is_device_capability_family(120)):
        return False
    if page_size is not None and page_size not in (32, 64, 128):
        return True
    if not envs.APHRODITE_USE_DEEP_GEMM:
        return True

    from aphrodite.utils.deep_gemm import _import_deep_gemm

    module = _import_deep_gemm()
    if module is None or not all(
        callable(getattr(module, name, None))
        for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits", "get_paged_mqa_logits_metadata")
    ):
        return True
    # The generic entry points also exist in revisions without SM120 kernels.
    include = Path(module.__file__).parent / "include" / "deep_gemm"
    if not all(
        (include / name).is_file()
        for name in (
            "impls/sm120_fp8_mqa_logits.cuh",
            "impls/sm120_fp8_paged_mqa_logits.cuh",
            "scheduler/sm120_paged_mqa_logits.cuh",
        )
    ):
        return True
    if page_size == 128:
        get_max_page_size = getattr(module, "get_sm120_fp8_paged_mqa_max_page_size", None)
        return not callable(get_max_page_size) or get_max_page_size() < page_size
    return False


@triton.jit
def _sm120_fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    starts_ptr,
    ends_ptr,
    logits_ptr,
    num_kv_tokens,
    stride_q_m: tl.int64,
    stride_q_h: tl.int64,
    stride_q_d: tl.int64,
    stride_k_n: tl.int64,
    stride_k_d: tl.int64,
    stride_w_m: tl.int64,
    stride_w_h: tl.int64,
    stride_o_m: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.arange(0, NUM_HEADS)[:, None]
    dims = tl.arange(0, HEAD_DIM)

    start = tl.maximum(tl.load(starts_ptr + row), 0)
    end = tl.minimum(tl.load(ends_ptr + row), num_kv_tokens)
    first_tile = start // BLOCK_KV
    last_tile = tl.cdiv(tl.maximum(end, 0), BLOCK_KV)

    q = tl.load(
        q_ptr + row * stride_q_m + heads * stride_q_h + dims[None, :] * stride_q_d,
    )
    weights = tl.load(weights_ptr + row * stride_w_m + heads * stride_w_h).to(tl.float32)
    for tile in range(first_tile + tl.program_id(1), last_tile, tl.num_programs(1)):
        kv = tile * BLOCK_KV + tl.arange(0, BLOCK_KV)
        valid = (kv >= start) & (kv < end) & (kv < num_kv_tokens)
        k = tl.load(
            k_ptr + dims[:, None] * stride_k_d + kv[None, :] * stride_k_n,
            mask=valid[None, :],
            other=0.0,
        )
        scale = tl.load(k_scale_ptr + kv, mask=valid, other=0.0).to(tl.float32)
        scores = tl.dot(q, k, input_precision="ieee").to(tl.float32)
        scores = tl.maximum(scores * scale[None, :], 0.0)
        scores = tl.sum(scores * weights, axis=0)
        scores = tl.where(valid, scores, -float("inf"))
        tl.store(logits_ptr + row * stride_o_m + kv, scores, mask=kv < num_kv_tokens)


@triton.jit
def _sm120_fp8_paged_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    max_model_len,
    stride_q_b: tl.int64,
    stride_q_n: tl.int64,
    stride_q_h: tl.int64,
    stride_q_d: tl.int64,
    stride_k_block: tl.int64,
    stride_k_token: tl.int64,
    stride_k_d: tl.int64,
    stride_s_block: tl.int64,
    stride_s_token: tl.int64,
    stride_ctx_b: tl.int64,
    stride_ctx_n: tl.int64,
    stride_bt_b: tl.int64,
    stride_bt_block: tl.int64,
    stride_w_m: tl.int64,
    stride_w_h: tl.int64,
    stride_o_m: tl.int64,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NEXT_N
    q_index = row % NEXT_N
    heads = tl.arange(0, NUM_HEADS)[:, None]
    dims = tl.arange(0, HEAD_DIM)
    context_len = tl.load(context_lens_ptr + batch * stride_ctx_b + q_index * stride_ctx_n)
    num_tiles = tl.cdiv(tl.maximum(tl.minimum(context_len, max_model_len), 0), BLOCK_KV)
    q = tl.load(
        q_ptr + batch * stride_q_b + q_index * stride_q_n + heads * stride_q_h + dims[None, :] * stride_q_d,
    )
    weights = tl.load(weights_ptr + row * stride_w_m + heads * stride_w_h).to(tl.float32)

    for tile in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        logical = tile * BLOCK_KV + tl.arange(0, BLOCK_KV)
        valid = (logical < context_len) & (logical < max_model_len)
        logical_block = logical // PAGE_SIZE
        block_offset = logical % PAGE_SIZE
        physical_block = tl.load(
            block_tables_ptr + batch * stride_bt_b + logical_block * stride_bt_block,
            mask=valid,
            other=0,
        )
        k = tl.load(
            k_ptr
            + physical_block[None, :] * stride_k_block
            + block_offset[None, :] * stride_k_token
            + dims[:, None] * stride_k_d,
            mask=valid[None, :],
            other=0.0,
        )
        scale = tl.load(
            k_scale_ptr + physical_block * stride_s_block + block_offset * stride_s_token,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(q, k, input_precision="ieee").to(tl.float32)
        scores = tl.maximum(scores * scale[None, :], 0.0)
        scores = tl.sum(scores * weights, axis=0)
        scores = tl.where(valid, scores, -float("inf"))
        tl.store(logits_ptr + row * stride_o_m + logical, scores, mask=logical < max_model_len)


def sm120_fp8_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Compute ragged FP8 scores, with -inf outside each row's [start, end)."""
    num_rows, num_heads, head_dim = q.shape
    num_kv_tokens = k.shape[0]
    logits = torch.full((num_rows, num_kv_tokens), -torch.inf, dtype=torch.float32, device=q.device)
    block_kv = 64
    parallel_tiles = min(128, triton.cdiv(num_kv_tokens, block_kv))
    _sm120_fp8_mqa_logits_kernel[(num_rows, parallel_tiles)](
        q,
        k,
        k_scales.reshape(-1),
        weights,
        starts,
        ends,
        logits,
        num_kv_tokens,
        *q.stride(),
        *k.stride(),
        *weights.stride(),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=2,
    )
    return logits


def _fp8_paged_cache_views(kv_cache: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    page_size, num_pages = kv_cache.shape[1], kv_cache.shape[0]
    page_stride = kv_cache.stride(0)
    cache_bytes = kv_cache.view(torch.uint8)
    # Each physical page stores all FP8 rows before its FP32 scales.
    k_bytes = torch.as_strided(
        cache_bytes,
        size=(num_pages, page_size, head_dim),
        stride=(page_stride, head_dim, 1),
    )
    scale_bytes = torch.as_strided(
        cache_bytes,
        size=(num_pages, page_size, 4),
        stride=(page_stride, 4, 1),
        storage_offset=cache_bytes.storage_offset() + page_size * head_dim,
    )
    return k_bytes.view(torch.float8_e4m3fn), scale_bytes.view(torch.float32).squeeze(-1)


def sm120_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute paged FP8 scores, with -inf after each query's context bound."""
    batch_size, next_n, num_heads, head_dim = q.shape
    page_size = kv_cache.shape[1]
    k, k_scales = _fp8_paged_cache_views(kv_cache, head_dim)
    if context_lens.ndim == 1:
        context_lens = context_lens[:, None].expand(-1, next_n)

    # Graphs retain the maximum width, but only live-context tiles need GEMMs.
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    block_kv = 64
    parallel_tiles = min(128, triton.cdiv(max_model_len, block_kv))
    _sm120_fp8_paged_mqa_logits_kernel[(batch_size * next_n, parallel_tiles)](
        q,
        k,
        k_scales,
        weights,
        context_lens,
        block_tables,
        logits,
        max_model_len,
        *q.stride(),
        *k.stride(),
        *k_scales.stride(),
        *context_lens.stride(),
        *block_tables.stride(),
        *weights.stride(),
        logits.stride(0),
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=2,
    )
    return logits
