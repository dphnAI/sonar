# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import torch.nn as nn

from aphrodite.model_executor.layers.fusion.quant_activation import QuantizedActivation
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
)
from aphrodite.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from aphrodite.platforms import current_platform
from aphrodite.utils.deep_gemm import fp8_einsum, get_tma_aligned_size, is_deep_gemm_supported


def _expand_block_scales(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from aphrodite.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        scale = _upcast_e8m0_to_fp32(scale)
    else:
        scale = scale.float()
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = (rows + row_blocks - 1) // row_blocks
    col_block = (cols + col_blocks - 1) // col_blocks
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    return torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]


def _bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Portable inverse-RoPE and dequantized WO_A fallback."""
    half_rope = rope_dim // 2
    rope = o[..., -rope_dim:].float()
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    cos = cos_sin_cache[positions, :half_rope].unsqueeze(1)
    sin = cos_sin_cache[positions, half_rope:].unsqueeze(1)
    inv_rope = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1).flatten(-2)
    o_ref = torch.cat((o[..., :-rope_dim].float(), inv_rope), dim=-1)
    o_ref = o_ref.view(o.shape[0], n_groups, -1)

    weight = getattr(wo_a, "_dsv4_wo_a_bf16", None)
    if weight is None:
        hidden_dim = heads_per_group * o.shape[-1]
        weight = wo_a.weight.view(n_groups, o_lora_rank, hidden_dim).float()
        weight_scale = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
        weight_scale = weight_scale.view(n_groups, -1, weight_scale.shape[-1])
        weight = (weight * _expand_block_scales(weight_scale, o_lora_rank, hidden_dim)).to(torch.bfloat16)
        wo_a._dsv4_wo_a_bf16 = weight
    return torch.einsum("tgd,grd->tgr", o_ref.to(torch.bfloat16), weight)


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor | QuantizedActivation,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization. ``wo_b`` is any callable over
    the flattened ``z``: the projection module itself, or a wrapper that also
    reduce-scatters its output (DeepSeek-V4.1 GEMM-RS).

    A QuantizedActivation ``o`` was already rotated and cast by the attention
    kernel, one ``wo_a`` group per slot with the live groups first, so only
    the padding groups are sliced off.
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    if use_fp8 and not is_deep_gemm_supported():
        assert not isinstance(o, QuantizedActivation)
        z = _bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
        return wo_b(z.flatten(1))

    if isinstance(o, QuantizedActivation):
        assert use_fp8
        o_proj_input, o_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
    else:
        o_proj_input, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=einsum_recipe[2],
            tma_aligned_scales=tma_aligned_scales,
            quantize=use_fp8,
        )
    z = torch.empty(
        (o_proj_input.shape[0], n_groups, o_lora_rank),
        device=o_proj_input.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        tokens = o_proj_input.shape[0]
        if (
            current_platform.is_device_capability_family(120)
            and 1 < tokens <= 64
            and o_proj_input.dtype == grouped_weight.dtype == torch.bfloat16
            and grouped_weight.is_contiguous()
            and o_lora_rank % 64 == 0
            and grouped_weight.shape[2] % (2048 if tokens > 32 else 1024) == 0
        ):
            from ._sm120_bf16_bmm import bf16_grouped_bmm

            bf16_grouped_bmm(o_proj_input, grouped_weight, z)
        else:
            torch.bmm(
                o_proj_input.transpose(0, 1),
                grouped_weight.transpose(1, 2),
                out=z.transpose(0, 1),
            )
    return wo_b(z.flatten(1))


# TRTLLM-gen's DSv4 RopeQuant epilogue (flashinfer-ai/flashinfer#4918) has one
# fixed schedule: 128 query heads in groups of 8 512-wide heads, and FP8 with
# one packed UE8M0 scale per 128 elements.
_ROPE_QUANT_NUM_HEADS = 128
_ROPE_QUANT_HEADS_PER_GROUP = 8
_ROPE_QUANT_HEAD_DIM = 512
_ROPE_QUANT_BLOCK = 128


def rope_quant_unsupported_reason(layer: torch.nn.Module) -> str | None:
    """Why ``layer`` cannot take FlashInfer's fused inverse RoPE + FP8 output.

    DSv4 always has 8 heads per ``wo_a`` group at any TP size, and the rest of
    the kernel's fixed shape (head/RoPE dims, per-128 einsum scales, FP32 RoPE
    cache) is what the SM100 layer uses anyway. What varies is the local head
    count and the configs the layer also serves unquantized.
    """
    if not is_deep_gemm_supported():
        return "RopeQuant needs DeepGEMM output projection support"
    if layer.n_local_heads != _ROPE_QUANT_NUM_HEADS:
        # Padding fewer heads up to the kernel's fixed 128 costs more
        # attention than the fused epilogue saves.
        return f"RopeQuant needs {_ROPE_QUANT_NUM_HEADS} local query heads"
    if layer.kv_cache_torch_dtype != torch.float8_e4m3fn:
        return "RopeQuant needs the per-tensor FP8 KV cache"
    if layer.wo_a.weight.dtype != torch.float8_e4m3fn:
        return "RopeQuant needs an FP8 wo_a"
    return None


def rope_quant_attn_out(num_tokens: int, device: torch.device) -> QuantizedActivation:
    """Allocate the output pair the RopeQuant kernel writes.

    Values are group-major, ``[num_tokens, 16, 8 * 512]`` over physical
    ``[group, token, K]``. Scales are MN-major packed UE8M0, one int32 per
    head over its four 128-wide blocks. Both are what ``fp8_einsum`` consumes.
    """
    n_groups = _ROPE_QUANT_NUM_HEADS // _ROPE_QUANT_HEADS_PER_GROUP
    group_width = _ROPE_QUANT_HEADS_PER_GROUP * _ROPE_QUANT_HEAD_DIM
    data = torch.empty((n_groups, num_tokens, group_width), dtype=torch.float8_e4m3fn, device=device).transpose(0, 1)
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    scale = torch.empty(
        (n_groups, group_width // (_ROPE_QUANT_BLOCK * 4), aligned),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:num_tokens]
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kFp8Dynamic128Sym,
    )
