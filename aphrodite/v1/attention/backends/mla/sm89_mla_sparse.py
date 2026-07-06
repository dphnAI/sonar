# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA attention backend for sm89 (Ada, e.g. RTX 4090) GPUs.

The stock DeepSeek sparse attention stack is kernel-gated to sm90+; this
backend drives the dedicated sm89 CUDA ops instead. The KV cache must use the
``fp8_ds_mla`` 656-byte-per-token layout (see flashmla_sparse for the format).
The forward kernel gathers rows directly from the fp8 pool by physical slot
for decode and prefill-extend alike, so there is no separate bf16 prefill
path and no upconvert workspace.
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
from aphrodite.logger import init_logger
from aphrodite.platforms import current_platform
from aphrodite.platforms.interface import DeviceCapability
from aphrodite.utils.math_utils import cdiv, round_up
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
from aphrodite.v1.attention.backends.utils import split_decodes_and_prefills
from aphrodite.v1.kv_cache_interface import AttentionSpec
from aphrodite.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from aphrodite.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


@cache
def use_sm89_dsa() -> bool:
    """Whether the sm89 DeepSeek sparse attention kernels should be used.

    The sparse-indexer logits and metadata paths key off this, so non-sm89
    builds are unaffected.
    """
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability((8, 9))
        and ops.supports_sm89_dsa()
        and not envs.APHRODITE_DISABLE_SM89_DSA
    )


class Sm89MLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # No bf16 cache path; the forward kernel reads the fp8_ds_mla pool
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
        # DeepSeek V3.2 layout, 512 NoPE + 64 RoPE = 576.
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
class Sm89LocalPrefillMetadata:
    """DCP local-prefill (fresh prompts only) metadata.

    Present iff every prefill request in the batch starts at context length 0.
    Every DCP rank then holds the complete prompt activations (replicated
    across the DCP group, which reuses the TP group), so each rank can build
    the prompt's full KV locally in a per-forward shadow pool and attend with
    its local heads, skipping the q all-gather, LSE combine, and indexer
    top-k merge for these tokens.
    """

    # int32 [num_prefill_tokens] per-token prefill request index (0-based
    # within the prefill segment); feeds the prefill-workspace branch of the
    # topk index conversion kernel.
    workspace_request_ids: torch.Tensor
    # int32 [num_prefills] row offset of each request's prompt in the shadow
    # pool (its query start within the prefill token segment).
    workspace_starts: torch.Tensor
    # int64 [num_prefill_tokens] identity slots (arange) for writing the
    # prefill tokens' fp8_ds_mla rows into the shadow pool.
    shadow_slot_mapping: torch.Tensor


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
    # Populated only under DCP when the local-prefill path is active; other
    # configurations leave these at their defaults.
    num_decode_tokens: int = 0
    prefill_local: Sm89LocalPrefillMetadata | None = None


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
        parallel_config = aphrodite_config.parallel_config
        # Local-prefill eligibility (static part). PCP shards prefill tokens
        # across ranks, breaking the premise that each rank holds the whole
        # prompt's activations; speculative decode changes the decode/prefill
        # split rules, so keep the gate simple and fall back.
        num_speculative_tokens = (
            aphrodite_config.speculative_config.num_speculative_tokens
            if aphrodite_config.speculative_config is not None
            else 0
        )
        self.allow_local_prefill = (
            parallel_config.decode_context_parallel_size > 1
            and parallel_config.prefill_context_parallel_size == 1
            and num_speculative_tokens == 0
        )
        if self.allow_local_prefill:
            # Identity slot mapping for the local-prefill shadow pool; prefill
            # token i (in batch order) lands in shadow row i.
            self.local_prefill_slot_buffer = torch.arange(
                aphrodite_config.scheduler_config.max_num_batched_tokens,
                dtype=torch.int64,
                device=device,
            )
        if parallel_config.decode_context_parallel_size > 1:
            self._warmup_dcp_kernels()

    def _warmup_dcp_kernels(self) -> None:
        """Pre-compile the Triton index-conversion kernel variants used under DCP.

        The startup profile/capture dummy runs never execute the sparse
        prefill branches (they carry no attention metadata), so without this
        the first real prefill pays the Triton JIT for each kernel variant,
        serialized across pipeline stages. Token count only enters the launch
        grid, never the compile key, so a single 1-token launch per variant
        covers every batch shape. All dummy indices are -1 (invalid), so the
        kernels write only -1 sentinels into throwaway buffers.
        """
        from aphrodite.distributed import get_dcp_group
        from aphrodite.v1.worker.cp_utils import get_total_cp_world_size

        try:
            device = self.device
            parallel_config = self.aphrodite_config.parallel_config
            block_size = self.kv_cache_spec.block_size
            num_topk = self.topk_tokens
            # Mirror the runner's persistent block-table width, which is part
            # of the compile key (constexpr max_num_blocks_per_req). See
            # MultiGroupBlockTable.__init__ for the width formula, including
            # the 128-token alignment round-up.
            width = cdiv(
                self.model_config.max_model_len,
                block_size * get_total_cp_world_size(),
            )
            if block_size <= 128:
                mult = 128 // block_size
                width = cdiv(width, mult) * mult
            req_id = torch.zeros(1, dtype=torch.int32, device=device)
            block_table = torch.zeros((1, width), dtype=torch.int32, device=device)
            indices = torch.full((1, num_topk), -1, dtype=torch.int32, device=device)
            # Local-prefill conversion (prefill-workspace branch); mirrors
            # forward_mqa_local_prefill.
            triton_convert_req_index_to_global_index(
                req_id,
                block_table,
                indices,
                BLOCK_SIZE=block_size,
                NUM_TOPK_TOKENS=num_topk,
                HAS_PREFILL_WORKSPACE=True,
                prefill_workspace_request_ids=req_id,
                prefill_workspace_starts=torch.zeros(1, dtype=torch.int32, device=device),
            )
            # Decode/mixed DCP filter + compact conversion; mirrors forward_mqa
            # (also warmed by CUDA graph capture, kept for enforce-eager runs).
            block_n = num_topk if num_topk & (num_topk - 1) == 0 else 128
            triton_filter_and_convert_dcp_index(
                req_id,
                block_table,
                indices,
                dcp_size=parallel_config.decode_context_parallel_size,
                dcp_rank=get_dcp_group().rank_in_group,
                cp_kv_cache_interleave_size=parallel_config.cp_kv_cache_interleave_size,
                BLOCK_SIZE=block_size,
                NUM_TOPK_TOKENS=num_topk,
                BLOCK_N=block_n,
                compact_valid_to_front=True,
                return_valid_counts=True,
            )
        except Exception:
            logger.warning(
                "sm89 DCP Triton kernel warmup failed; kernels will be JIT-compiled on first use instead.",
                exc_info=True,
            )

    def _build_local_prefill(
        self,
        cm: CommonAttentionMetadata,
        seg_lengths: np.ndarray,
        starts: np.ndarray,
    ) -> tuple[int, "Sm89LocalPrefillMetadata | None"]:
        """Local-prefill metadata for DCP, or (0, None) if inapplicable.

        Active only when every prefill request is a fresh prompt (context
        length 0), so the whole prompt's KV is computable on-rank from the
        replicated activations. Continuation chunks (context > 0) fall back to
        the existing full-DCP path for the entire batch, since the gate must
        be batch-uniform for every DCP rank to issue the same collectives.
        """
        if not self.allow_local_prefill:
            return 0, None
        # Must match the indexer builder's split exactly (same threshold and
        # require_uniform) so both sides agree on the decode/prefill boundary.
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            cm,
            decode_threshold=self.reorder_batch_threshold or 1,
            require_uniform=True,
        )
        if num_prefills == 0:
            return num_decode_tokens, None
        # Exact for prefill rows (see CommonAttentionMetadata docstring).
        seq_lens_cpu = cm.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        prefill_query_lens = seg_lengths[num_decodes:]
        all_fresh = np.array_equal(
            np.asarray(seq_lens_cpu[num_decodes:], dtype=np.int64),
            prefill_query_lens.astype(np.int64),
        )
        if not all_fresh:
            # Expected for continuation chunks (chunked prompts, preemption
            # resumes, prefix-cache partial hits); logged once so a fast path
            # that never engages (e.g. a gating regression) is visible.
            logger.info_once(
                "DCP local-prefill fast path inactive for this batch "
                "(non-fresh prefill present); using the full-DCP prefill path."
            )
            return num_decode_tokens, None

        ws_starts = (starts[num_decodes:-1] - num_decode_tokens).astype(np.int32)
        ws_req_ids = np.repeat(np.arange(num_prefills, dtype=np.int32), prefill_query_lens)
        workspace_starts = torch.empty(num_prefills, dtype=torch.int32, device=self.device)
        workspace_starts.copy_(np_to_pinned_tensor(ws_starts), non_blocking=True)
        workspace_request_ids = torch.empty(num_prefill_tokens, dtype=torch.int32, device=self.device)
        workspace_request_ids.copy_(np_to_pinned_tensor(ws_req_ids), non_blocking=True)
        return num_decode_tokens, Sm89LocalPrefillMetadata(
            workspace_request_ids=workspace_request_ids,
            workspace_starts=workspace_starts,
            shadow_slot_mapping=self.local_prefill_slot_buffer[:num_prefill_tokens],
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

        num_decode_tokens, prefill_local = self._build_local_prefill(cm, seg_lengths, starts)

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
            cp_kv_cache_interleave_size=(self.aphrodite_config.parallel_config.cp_kv_cache_interleave_size),
            num_decode_tokens=num_decode_tokens,
            prefill_local=prefill_local,
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
        dcp_world_size = aphrodite_config.parallel_config.decode_context_parallel_size
        self.prefill_local_pool: torch.Tensor | None = None
        self.prefill_local_pool_paged: torch.Tensor | None = None
        if dcp_world_size > 1:
            # DCP local-prefill shadow pool, one fp8_ds_mla row (656 B) per
            # prefill token, written each forward with an identity slot
            # mapping. Rounded up to the 64-token cache block so it can be
            # viewed with the paged shape concat_and_cache_mla expects; with
            # identity slots the paged view is exactly the flat row view.
            pool_rows = round_up(max_tokens, 64)
            self.q_concat_buffer, self.prefill_local_pool = current_workspace_manager().get_simultaneous(
                ((max_tokens, num_heads, head_size), torch.bfloat16),
                ((pool_rows, 656), torch.uint8),
            )
            self.prefill_local_pool_paged = self.prefill_local_pool.view(pool_rows // 64, 64, 656)
        else:
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
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        topk_lens: torch.Tensor | None = None
        if self.dcp_world_size > 1:
            # Keep only the slots this DCP rank owns, mapped to local physical
            # cache slots and compacted to a per-row prefix (tail -1). The
            # per-token valid counts go to the kernel as topk_lens so it stops
            # after the owned prefix (~topk/dcp_world_size slots) instead of
            # scanning all topk candidates. BLOCK_N == topk compacts each row
            # in a single tile, which keeps the prefix in column order and the
            # pass run-to-run deterministic (the multi-tile atomic slot
            # allocator would not); fall back to the default tile width if
            # topk is not a power of two (tl.arange constraint).
            num_topk = topk_indices.shape[1]
            block_n = num_topk if num_topk & (num_topk - 1) == 0 else 128
            # Slice req_id_per_token to the tokens covered by q; under the DCP
            # local-prefill split q holds only the leading decode tokens (the
            # conversion grid is sized from this tensor). A full q makes this
            # slice a no-op.
            topk_indices, topk_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=num_topk,
                BLOCK_N=block_n,
                compact_valid_to_front=True,
                return_valid_counts=True,
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
        # topk_lens (like the converted indices) is allocated per call, so
        # under CUDA-graph capture it lives in the graph pool and the captured
        # conversion kernels rewrite it on every replay, same as attn_out/lse.
        ops.sm89_sparse_mla_fwd(q, kv_pool, topk_indices, attn_out, lse, self.softmax_scale, topk_lens)
        return attn_out, lse if self.need_to_return_lse_for_decode else None

    def build_local_prefill_pool(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: Sm89MLASparseMetadata,
        k_scale: torch.Tensor,
    ) -> None:
        """Quantize this batch's fresh-prompt KV into the local shadow pool.

        Uses the same concat_and_cache_mla fp8_ds_mla kernel as the real cache
        write, so shadow rows are numerically identical to what the DCP path
        would read back from the sharded cache. Identity slots, prefill token
        i (batch order) -> shadow row i. Runs on every DCP rank with
        identical, replicated inputs; no collective involved.
        """
        prefill_local = attn_metadata.prefill_local
        assert prefill_local is not None
        assert self.prefill_local_pool_paged is not None
        num_decode_tokens = attn_metadata.num_decode_tokens
        ops.concat_and_cache_mla(
            kv_c_normed[num_decode_tokens:],
            k_pe[num_decode_tokens:].squeeze(1),
            self.prefill_local_pool_paged,
            prefill_local.shadow_slot_mapping,
            kv_cache_dtype="fp8_ds_mla",
            scale=k_scale,
        )

    def forward_mqa_local_prefill(
        self,
        q: torch.Tensor,
        attn_metadata: Sm89MLASparseMetadata,
    ) -> torch.Tensor:
        """Collective-free sparse attention for fresh-prompt prefill tokens.

        q carries only this rank's local heads (no DCP all-gather). The
        indexer produced per-request global token positions (its DCP merge is
        skipped in this mode), which map affinely into the shadow pool via the
        prefill-workspace branch of the conversion kernel. Since the shadow
        pool holds the complete prompt, single-pass local attention is exact
        and no LSE combine is needed.
        """
        prefill_local = attn_metadata.prefill_local
        assert prefill_local is not None
        assert self.prefill_local_pool is not None
        assert self.topk_indices_buffer is not None
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = q.shape[0]
        token_slice = slice(num_decode_tokens, num_decode_tokens + num_prefill_tokens)

        topk_indices = self.topk_indices_buffer[token_slice]
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[token_slice],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            HAS_PREFILL_WORKSPACE=True,
            prefill_workspace_request_ids=prefill_local.workspace_request_ids,
            prefill_workspace_starts=prefill_local.workspace_starts,
        )

        num_heads = q.shape[1]
        attn_out = q.new_empty((num_prefill_tokens, num_heads, self.kv_lora_rank))
        lse = q.new_empty((num_prefill_tokens, num_heads), dtype=torch.float32)
        ops.sm89_sparse_mla_fwd(
            q,
            self.prefill_local_pool,
            topk_indices,
            attn_out,
            lse,
            self.softmax_scale,
            None,
        )
        return attn_out
