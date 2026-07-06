# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA attention backend for sm89 (Ada, e.g. RTX 4090) GPUs.

The stock DeepSeek sparse attention stack is kernel-gated to sm90+; this
backend drives the dedicated sm89 CUDA ops instead. The KV cache must use the
``fp8_ds_mla`` 656-byte-per-token layout (see flashmla_sparse for the format
description): the forward kernel gathers rows directly from the fp8 pool by
physical slot for decode and prefill-extend alike, so there is no separate
bf16 prefill path and no upconvert workspace.
"""

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

import aphrodite.envs as envs
from aphrodite import _custom_ops as ops
from aphrodite.config import AphroditeConfig, get_current_aphrodite_config
from aphrodite.config.cache import CacheDType
from aphrodite.platforms import current_platform
from aphrodite.platforms.interface import DeviceCapability
from aphrodite.utils.torch_utils import np_to_pinned_tensor
from aphrodite.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from aphrodite.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from aphrodite.v1.kv_cache_interface import AttentionSpec
from aphrodite.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from aphrodite.model_executor.models.deepseek_v2 import Indexer


@cache
def use_sm89_dsa() -> bool:
    """Whether the sm89 DeepSeek sparse attention kernels should be used.

    True only on an sm89 CUDA device with the kernels compiled in and the
    APHRODITE_DISABLE_SM89_DSA kill switch unset; the sparse-indexer logits
    and metadata paths key off this so a non-sm89 build is unaffected.
    """
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability((8, 9))
        and ops.supports_sm89_dsa()
        and not envs.APHRODITE_DISABLE_SM89_DSA
    )


class Sm89MLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # No bf16 cache path: the forward kernel reads the fp8_ds_mla pool
    # directly, so `--kv-cache-dtype fp8_ds_mla` (or `fp8`) is required.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "SM89_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["Sm89MLASparseMetadataBuilder"]:
        return Sm89MLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[SparseMLAAttentionImpl[Any]]:
        return Sm89MLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V3.2 layout: 512 NoPE + 64 RoPE = 576.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) == (8, 9)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if envs.APHRODITE_DISABLE_SM89_DSA:
            return "SM89 MLA Sparse is disabled via APHRODITE_DISABLE_SM89_DSA"

        if not ops.supports_sm89_dsa():
            return "sm89 DSA kernels are not compiled into this build"

        from aphrodite.config import get_current_aphrodite_config_or_none

        aphrodite_config = get_current_aphrodite_config_or_none()
        if aphrodite_config is not None and aphrodite_config.model_config is not None:
            hf_config = aphrodite_config.model_config.hf_config
            if not hasattr(hf_config, "index_topk"):
                return "SM89 MLA Sparse requires model with index_topk"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # 656-byte custom storage format; see flashmla_sparse docstring.
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)


@dataclass
class Sm89MLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1


class Sm89MLASparseMetadataBuilder(AttentionMetadataBuilder[Sm89MLASparseMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        aphrodite_config: AphroditeConfig,
        device: torch.device,
    ) -> None:
        self.aphrodite_config = aphrodite_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = aphrodite_config.model_config
        self.device = device

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        self.topk_tokens = aphrodite_config.model_config.hf_config.index_topk
        self.req_id_per_token_buffer = torch.empty(
            (aphrodite_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Sm89MLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths)
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            np_to_pinned_tensor(req_id_per_token), non_blocking=True
        )

        return Sm89MLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            cp_kv_cache_interleave_size=(
                self.aphrodite_config.parallel_config.cp_kv_cache_interleave_size
            ),
        )


class Sm89MLASparseImpl(SparseMLAAttentionImpl[Sm89MLASparseMetadata]):
    # The kernel emits a natural-log LSE (lse_base_on_e default), which the
    # DCP cross-rank softmax combine consumes.
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args: Any,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError("Sm89MLASparseImpl does not support alibi, sliding window, or logits soft cap.")
        if kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "SM89 MLA Sparse requires the fp8_ds_mla KV cache layout; launch with --kv-cache-dtype fp8_ds_mla."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = float(scale)
        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        assert self.topk_indices_buffer is not None, "Indexer or topk_indices_buffer required for sparse MLA"
        # The kernel takes a bf16 query and dequantizes KV internally.
        self.supports_quant_query_input = False

        aphrodite_config = get_current_aphrodite_config()
        max_tokens = aphrodite_config.scheduler_config.max_num_batched_tokens
        (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
            ((max_tokens, num_heads, head_size), torch.bfloat16),
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: Sm89MLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        if self.dcp_world_size > 1:
            # Keep only the slots this DCP rank owns, mapped to local physical
            # cache slots; non-owned entries become -1. The kernel masks
            # interior -1s natively, so no compaction or valid-count side
            # channel is needed.
            topk_indices = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                compact_valid_to_front=False,
            )
        else:
            # Per-request token indices -> physical cache slots. -1 padding
            # stays -1; the kernel zero-outputs fully masked rows, so no
            # valid-count side channel is needed.
            topk_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )

        # The kernel addresses the cache as raw per-token rows ([slots, 656]).
        kv_pool = kv_c_and_k_pe_cache.view(torch.uint8)
        kv_pool = kv_pool.view(-1, kv_pool.shape[-1])

        # Under DCP the caller all-gathers q across the group in the head dim,
        # so q carries num_heads * dcp_world_size heads; size outputs from q.
        num_heads = q.shape[1]
        attn_out = q.new_empty((num_actual_toks, num_heads, self.kv_lora_rank))
        lse = q.new_empty((num_actual_toks, num_heads), dtype=torch.float32)
        ops.sm89_sparse_mla_fwd(q, kv_pool, topk_indices, attn_out, lse, self.softmax_scale)
        return attn_out, lse if self.need_to_return_lse_for_decode else None
