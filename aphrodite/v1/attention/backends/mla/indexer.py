# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch

import aphrodite.envs as envs
from aphrodite import _custom_ops as ops
from aphrodite.config import AphroditeConfig
from aphrodite.distributed import get_dcp_group
from aphrodite.logger import init_logger
from aphrodite.platforms import current_platform
from aphrodite.triton_utils import tl, triton
from aphrodite.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
)
from aphrodite.utils.math_utils import cdiv
from aphrodite.utils.platform_utils import num_compute_units
from aphrodite.utils.torch_utils import np_to_pinned_tensor
from aphrodite.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from aphrodite.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from aphrodite.v1.attention.backends.mla.sm89_mla_sparse import use_sm89_dsa
from aphrodite.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from aphrodite.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from aphrodite.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # Compute number of KVs attended to by this token.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = seq_len - max_decode_len + local_idx + 1
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # Copy block table row.
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # All reqs now have decode_len = 1.
    tl.store(decode_lens_ptr + idx, 1)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class DeepseekV32IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, 64] if current_platform.is_rocm() else [64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # DeepseekV32Indexer kernels do not support cross-layer
            # KV cache layout. Identity permutation keeps num_layers
            # first, signaling incompatibility.
            return (0, 1, 2, 3)
        return (0, 1, 2)


class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    # Under DCP (dcp_world_size > 1) these hold this rank's local row bounds;
    # otherwise they hold the global bounds.
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False
    local_cu_seq_lens: torch.Tensor | None = None
    local_total_seq_lens: int = 0
    max_local_total_seq_lens: int = 0


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]
    # sm89 DCP local-prefill mode (fresh prompts only). Chunk metadata was
    # built with global (dcp=1) row bounds, chunk.block_table points into a
    # per-forward shadow K cache quantized locally from this batch's k, and
    # the cross-rank top-k merge is skipped. See Sm89LocalPrefillMetadata.
    dcp_local_prefill: bool = False
    # int64 [num_prefill_tokens] block-aligned identity slots for writing the
    # prefill tokens' quantized K into the shadow cache.
    shadow_slot_mapping: torch.Tensor | None = None
    # Number of (block_size-token) pages in the shadow cache.
    shadow_num_pages: int = 0


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_fp4_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    global_seq_lens: torch.Tensor | None = None


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


def get_max_prefill_buffer_size(aphrodite_config: AphroditeConfig):
    max_model_len = aphrodite_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    # The indexer opts out of the shared reorder-threshold vote (see __init__),
    # so this is None; its own split uses self.decode_threshold.
    reorder_batch_threshold: int | None = None

    @classmethod
    def get_cudagraph_support(
        cls,
        aphrodite_config: AphroditeConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.aphrodite_config.scheduler_config
        parallel_config = self.aphrodite_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        # The DCP sparse-indexer code is parameterized by interleave size, but
        # interleave > 1 is not yet validated end-to-end (gsm8k parity fails),
        # so fail closed here rather than silently produce wrong output.
        if self.dcp_world_size > 1 and self.cp_kv_cache_interleave_size > 1:
            raise NotImplementedError(
                "DCP sparse indexer currently supports only "
                f"cp_kv_cache_interleave_size=1 (got "
                f"{self.cp_kv_cache_interleave_size})."
            )
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.aphrodite_config)
        self.num_speculative_tokens = (
            self.aphrodite_config.speculative_config.num_speculative_tokens
            if self.aphrodite_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = self.aphrodite_config.attention_config.use_fp4_indexer_cache

        assert current_platform.is_device_capability_family(100) or not self.use_fp4_indexer_cache, (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )

        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        self.reorder_batch_threshold = None
        # NOTE: SM100 datacenter GPUs support any next_n natively via the
        # multi-atom paged MQA logits kernels (FP8 and FP4 indexer
        # caches). Outside the SM100 family the FP8
        # paged MQA logits kernel only supports next_n in (1, 2)
        # (deepgemm smxx_fp8_fp4_paged_mqa_logits.hpp:233), so flatten there.
        self.use_flattening = not current_platform.is_device_capability_family(100) and next_n not in (1, 2)
        logger.info_once(
            "DSA indexer decode path: use_flattening=%s (next_n=%d, use_fp4_indexer_cache=%s)",
            self.use_flattening,
            next_n,
            self.use_fp4_indexer_cache,
        )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count
        self.use_sm89_dsa = use_sm89_dsa()
        if self.use_sm89_dsa:
            # The sm89 paged-logits kernel runs a fixed persistent-CTA grid
            # (2 CTAs/SM x 128 SMs) and derives its partition count P from the
            # (P+1, 2) scheduler table, so size the buffer for P=256 instead of
            # the SM count.
            self.num_sms = 256

        self.offsets_buffer = torch.arange(next_n, device=self.device, dtype=torch.int32)
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # Shared workspace for decode seq_lens. Native MTP views this as
        # (B, max_decode_len) at runtime, keeping context_lens contiguous even
        # when max_decode_len is smaller than next_n.
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.global_decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.aphrodite_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty((self.num_sms + 1, 2), dtype=torch.int32, device=self.device)

        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio
        if self.dcp_world_size > 1 and self.compress_ratio > 1:
            raise NotImplementedError(
                f"DCP is not supported with sparse indexer KV compression (compress_ratio={self.compress_ratio})."
            )

        if self.dcp_world_size > 1 and self.use_sm89_dsa:
            self._warmup_dcp_kernels()

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (DeepseekV4)
            # Compressed slot mapping output buffer
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

    def _warmup_dcp_kernels(self) -> None:
        """Pre-compile the JIT kernels used by the DCP indexer prefill paths.

        The startup profile/capture dummy runs never reach the real indexer
        branches (no attention metadata), so the first real prefill would
        otherwise pay the Triton compiles (and the first continuation-chunk
        prefill the multi-second CuteDSL top-k merge compile), serialized
        across pipeline stages. Warmed with 1-request/1-token dummies; the
        compile keys never include the token count, so one launch per variant
        covers every batch shape.
        """
        from aphrodite.utils.import_utils import has_cutedsl

        try:
            device = self.device
            qsl_cpu = torch.tensor([0, 1], dtype=torch.int32)
            qsl = qsl_cpu.to(device)
            ones_cpu = torch.ones(1, dtype=torch.int32)
            ones = ones_cpu.to(device)
            bt = torch.zeros((1, 1), dtype=torch.int32, device=device)
            # One launch compiles the chunk-metadata kernel for all shapes
            # (its per-shape scalars are do_not_specialize) and both the
            # local-prefill (dcp=1) and continuation (dcp=world) launches.
            build_prefill_chunk_metadata(
                0,
                1,
                qsl,
                qsl_cpu,
                ones,
                ones,
                ones_cpu,
                bt,
                self.compress_ratio,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )

            # Cross-rank top-k merge (continuation-chunk prefill and decode).
            index_topk = getattr(self.aphrodite_config.model_config.hf_config, "index_topk", None)
            if has_cutedsl() and index_topk in (512, 1024, 2048):
                from aphrodite.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
                    StableTopKFromGatheredCandidatesKernel,
                    pack_dcp_topk_candidates_cutedsl,
                )

                logits = torch.zeros((1, 1), dtype=torch.float32, device=device)
                topk_idx = torch.full((1, index_topk), -1, dtype=torch.int32, device=device)
                packed = torch.empty((1, index_topk, 2), dtype=torch.float32, device=device)
                row_starts = torch.zeros(1, dtype=torch.int32, device=device)
                # Prefill merge passes row_starts, decode merge does not
                # (HAS_ROW_STARTS is part of the compile key); warm both.
                for rs in (row_starts, None):
                    pack_dcp_topk_candidates_cutedsl(
                        logits,
                        topk_idx,
                        packed,
                        self.dcp_rank,
                        self.dcp_world_size,
                        self.cp_kv_cache_interleave_size,
                        rs,
                    )
                # Keyed on (topk, num_candidates) with a symbolic row count,
                # so one compile covers every batch. This is the seconds-scale
                # CuteDSL compile.
                StableTopKFromGatheredCandidatesKernel.compile(index_topk, self.dcp_world_size * index_topk)
        except Exception:
            logger.warning(
                "DCP indexer kernel warmup failed; kernels will be JIT-compiled on first use instead.",
                exc_info=True,
            )

    def _use_dcp_local_prefill(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
    ) -> bool:
        """Whether this batch qualifies for the sm89 DCP local-prefill path.

        Fresh prompts only (context length 0 for every prefill request). The
        full prompt K is then computable on-rank from the replicated
        activations, so top-k can be selected over the local full-prompt
        logits and the cross-rank merge skipped. Continuation chunks fall back
        to the existing sharded-gather + merge path for the whole batch. The
        decision uses batch-level CPU metadata identical on all DCP ranks, so
        collective participation stays aligned. Must mirror the gating in
        Sm89MLASparseMetadataBuilder._build_local_prefill.
        """
        parallel_config = self.aphrodite_config.parallel_config
        if (
            not self.use_sm89_dsa
            or self.dcp_world_size <= 1
            or parallel_config.prefill_context_parallel_size > 1
            or self.num_speculative_tokens > 0
            or num_prefills == 0
        ):
            return False
        prefill_seq_lens = np.asarray(seq_lens_cpu[num_decodes:], dtype=np.int64)
        prefill_query_lens = np.asarray(prefill_query_lens_cpu, dtype=np.int64)
        return np.array_equal(prefill_seq_lens, prefill_query_lens)

    def _build_local_prefill_shadow_tensors(
        self,
        prefill_query_lens_cpu: torch.Tensor,
        num_prefill_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Identity block table + block-aligned slots for the shadow K cache.

        Each prefill request gets a contiguous, page-aligned span of the
        shadow cache (page alignment is required because the paged layout
        interleaves per-page scale blocks). Returns (identity_block_table
        [num_prefills, max_pages] int32, slot_mapping [num_prefill_tokens]
        int64, num_pages).
        """
        block = self.kv_cache_spec.block_size
        lens = np.asarray(prefill_query_lens_cpu, dtype=np.int64)
        pages_per_req = (lens + block - 1) // block
        page_starts = np.concatenate(([0], np.cumsum(pages_per_req)))
        shadow_num_pages = int(page_starts[-1])
        max_pages = int(pages_per_req.max())
        ibt = (page_starts[:-1, None] + np.arange(max_pages, dtype=np.int64)[None, :]).astype(np.int32)
        token_starts = np.concatenate(([0], np.cumsum(lens)))[:-1]
        slots = (
            np.repeat(page_starts[:-1] * block - token_starts, lens) + np.arange(num_prefill_tokens, dtype=np.int64)
        ).astype(np.int64)

        ibt_gpu = torch.empty(ibt.shape, dtype=torch.int32, device=self.device)
        ibt_gpu.copy_(np_to_pinned_tensor(ibt), non_blocking=True)
        slots_gpu = torch.empty(slots.shape, dtype=torch.int64, device=self.device)
        slots_gpu.copy_(np_to_pinned_tensor(slots), non_blocking=True)
        return ibt_gpu, slots_gpu, shadow_num_pages

    def _dcp_localize_decode_seq_lens(
        self,
        seq_lens: torch.Tensor,
        num_decodes: int,
        seq_lens_is_buffer_view: bool,
    ) -> torch.Tensor:
        local_seq_lens = get_dcp_local_seq_lens(
            seq_lens,
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )
        if seq_lens_is_buffer_view:
            seq_lens.copy_(local_seq_lens)
            return seq_lens

        out = self.decode_seq_lens_buffer[:num_decodes]
        out.copy_(local_seq_lens)
        return out

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Expand seq_lens/block_table/decode_lens for the decode kernels.

        Flatten path (not use_native, max_decode_len > 1):
          Each multi-token decode request is expanded into individual
          single-token entries so the kernel always sees next_n=1.

        Native path (use_native or max_decode_len == 1):
          Plain decode or spec-decode with 2D per-token context lengths.

        Returns (seq_lens, block_table, decode_lens, batch_size, requires_padding).
        seq_lens is 1D (batch_size,) for flatten/plain, 2D (B, max_decode_len)
        for native MTP.
        """
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            if min_decode_len == max_decode_len:
                # Uniform decode lengths.
                num_decode_tokens = num_decodes * max_decode_len
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                expanded_offsets = torch.repeat_interleave(
                    seq_lens - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.decode_seq_lens_buffer[:actual_expanded] = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = torch.repeat_interleave(
                    block_table, decode_lens, dim=0, output_size=actual_expanded
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[actual_expanded:num_decode_tokens, 0] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len): token j attends to
                # L - max_decode_len + j + 1 KV tokens.
                seq_lens_buffer = self.decode_seq_lens_buffer[: num_decodes * max_decode_len].view(
                    num_decodes, max_decode_len
                )
                seq_lens_buffer[:] = seq_lens.unsqueeze(1) - max_decode_len + 1 + self.offsets_buffer[:max_decode_len]
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def _prepare_global_decode_seq_lens(
        self,
        global_seq_lens: torch.Tensor | None,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decode_tokens: int,
        use_native: bool,
        max_decode_len: int,
    ) -> torch.Tensor | None:
        if global_seq_lens is None:
            return None
        if use_native or max_decode_len <= 1:
            return global_seq_lens

        actual_expanded = int(decode_lens_cpu.sum().item())
        if actual_expanded > 0:
            expanded_offsets = torch.repeat_interleave(
                global_seq_lens - decode_lens - query_start_loc,
                decode_lens,
                output_size=actual_expanded,
            )
            self.global_decode_seq_lens_buffer[:actual_expanded] = (
                expanded_offsets + self.arange_buffer[:actual_expanded] + 1
            )
        self.global_decode_seq_lens_buffer[actual_expanded:num_decode_tokens] = 0
        return self.global_decode_seq_lens_buffer[:num_decode_tokens]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            require_uniform=not self.use_flattening,
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        if self.compress_ratio > 1:
            block_table.clamp_(min=0)
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
            compressed_seq_lens = seq_lens // self.compress_ratio

        prefill_metadata = None
        if num_prefills > 0:
            # This CPU value is an upper bound for async-spec extend rows.  It
            # is safe for chunking/allocation because CUDA metadata below is
            # built from exact device seq_lens and gather ignores the tail.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            compressed_seq_lens_cpu = seq_lens_cpu // self.compress_ratio if self.compress_ratio > 1 else seq_lens_cpu
            prefill_query_lens_cpu = torch.diff(query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1])
            max_logits_bytes = envs.APHRODITE_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below).
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )

            # sm89 DCP local-prefill builds chunk metadata with global (dcp=1)
            # row bounds so top-k runs over each rank's local full-prompt K
            # (quantized from this batch's replicated k into the shadow cache)
            # and the cross-rank merge is skipped.
            local_prefill = self._use_dcp_local_prefill(num_decodes, num_prefills, seq_lens_cpu, prefill_query_lens_cpu)
            shadow_block_table = None
            shadow_slot_mapping = None
            shadow_num_pages = 0
            if local_prefill:
                shadow_block_table, shadow_slot_mapping, shadow_num_pages = self._build_local_prefill_shadow_tensors(
                    prefill_query_lens_cpu, num_prefill_tokens
                )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    self.compress_ratio,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                    dcp_rank=0 if local_prefill else self.dcp_rank,
                    dcp_world_size=1 if local_prefill else self.dcp_world_size,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    if local_prefill:
                        assert shadow_block_table is not None
                        # Gather this chunk's K from the shadow cache instead
                        # of the (sharded) real cache.
                        metadata.block_table = shadow_block_table[
                            req_slice.start - num_decodes : req_slice.stop - num_decodes
                        ]
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(
                chunks,
                dcp_local_prefill=local_prefill,
                shadow_slot_mapping=shadow_slot_mapping,
                shadow_num_pages=shadow_num_pages,
            )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu[: num_decodes + 1])

            # Under DCP the per-token decode bounds must be localized AFTER the
            # per-token expansion below, not before. Expanding from a
            # request-level localized length subtracts decode offsets in local
            # space and yields too-short bounds (e.g. world=2, rank=1, global
            # per-token bounds [8, 9, 10] -> [3, 4, 5] instead of [4, 4, 5]), so
            # the first decode token would run top-k against too short a local KV
            # range and miss valid tokens. Keep the global seq_lens here and
            # localize the expanded bounds further down.
            global_seq_lens_for_decode: torch.Tensor | None = None
            if dcp_local_seq_lens is not None:
                global_seq_lens_for_decode = common_attn_metadata.seq_lens[:num_decodes]
            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len <= next_n

            global_seq_lens_for_decode = self._prepare_global_decode_seq_lens(
                global_seq_lens=global_seq_lens_for_decode,
                decode_lens=decode_lens,
                decode_lens_cpu=decode_lens_cpu,
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                num_decode_tokens=num_decode_tokens,
                use_native=use_native,
                max_decode_len=max_decode_len,
            )

            seq_lens, block_table, decode_lens, batch_size, requires_padding = self._prepare_decode_tensors(
                seq_lens=seq_lens,
                block_table=block_table,
                decode_lens=decode_lens,
                decode_lens_cpu=decode_lens_cpu,
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                use_native=use_native,
                next_n=next_n,
                max_decode_len=max_decode_len,
            )

            seq_lens_is_buffer_view = (use_native and next_n > 1) or (not use_native and max_decode_len > 1)

            # DCP: localize the now-expanded per-token global bounds to this
            # rank's owned KV. Done here (after expansion) so each token's global
            # causal length is localized individually; see the comment above.
            if dcp_local_seq_lens is not None:
                seq_lens = self._dcp_localize_decode_seq_lens(seq_lens, num_decodes, seq_lens_is_buffer_view)

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                if seq_lens_is_buffer_view:
                    seq_lens //= self.compress_ratio
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    self.expanded_seq_lens_buffer[:num_decodes] = seq_lens // self.compress_ratio
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
            # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
            # kernels see the same (B, next_n) layout as the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            # Must be checked before the deep_gemm branch; deep_gemm may be
            # pip-installed on sm89 even though its kernels cannot run there.
            if self.use_sm89_dsa:
                # Fills the persistent (P+1, 2) partition table in place, so
                # the buffer address stays stable across CUDA graph replays.
                ops.sm89_paged_mqa_logits_metadata(
                    seq_lens,
                    self.scheduler_metadata_buffer,
                    seq_lens.shape[1],
                )
            # DeepGEMM is required for the paged MQA logits on CUDA devices
            elif current_platform.is_cuda() and has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
                global_seq_lens=global_seq_lens_for_decode,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    local_cu_seq_lens = cu_seq_lens
    local_total_seq_lens = total_seq_lens
    max_local_total_seq_lens = total_seq_lens
    if dcp_world_size > 1:
        # Per-rank local KV length under interleave-aware DCP sharding, shape
        # [num_reqs, dcp_world_size]. Reuse the canonical CP helper so the
        # sharding matches the rest of the DCP pipeline (decode/prefill).
        local_seq_lens = get_dcp_local_seq_lens(
            compressed_seq_lens[start_idx:end_idx],
            dcp_world_size,
            None,
            cp_kv_cache_interleave_size,
        )
        this_rank_counts = local_seq_lens[:, dcp_rank].to(torch.int32)
        local_cu_seq_lens = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        torch.cumsum(this_rank_counts, dim=0, out=local_cu_seq_lens[1:])
        local_total_seq_lens = int(local_cu_seq_lens[-1].item())
        max_local_total_seq_lens = int(local_seq_lens.sum(dim=0).max().item())

    query_start_loc = query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]

    total_query_len = int((query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item())
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    # Under DCP the kernel writes this rank's local row bounds into
    # cu_seq_len_ks/ke; otherwise local_cu_seq_lens aliases cu_seq_lens.
    _build_prefill_chunk_metadata_kernel[(num_reqs,)](
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        local_cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
        local_cu_seq_lens=local_cu_seq_lens,
        local_total_seq_lens=local_total_seq_lens,
        max_local_total_seq_lens=max_local_total_seq_lens,
    )


# The query-slice bounds vary with every batch's chunking (and the DCP
# scalars are per-rank constants); keep them out of the specialization key so
# the kernel compiles once instead of re-JITting per {==1, %16==0, other}
# value bucket as prefill shapes change.
@triton.jit(
    do_not_specialize=[
        "query_slice_start",
        "query_slice_stop",
        "DCP_RANK",
        "DCP_WORLD",
        "DCP_INTERLEAVE",
    ]
)
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
    uncompressed_seq_lens_ptr,
    cu_compressed_seq_lens_ptr,
    # Row-start base for cu_seq_len_ks/ke: local cumulative lens under DCP,
    # aliases cu_compressed_seq_lens_ptr otherwise.
    row_start_cu_compressed_seq_lens_ptr,
    # Outputs
    token_to_seq_ptr,
    cu_compressed_seq_len_ks_ptr,
    cu_compressed_seq_len_ke_ptr,
    query_slice_start,
    query_slice_stop,
    DCP_RANK,
    DCP_WORLD,
    DCP_INTERLEAVE,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
    compressed_seq_len = seq_end - seq_start

    # Row start for the (possibly localized) cu_seq_len_ks/ke. Equals seq_start
    # when DCP is disabled (the pointer aliases cu_compressed_seq_lens_ptr).
    row_start = tl.load(row_start_cu_compressed_seq_lens_ptr + batch_idx)

    uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
    start_pos = uncompressed_seq_len - query_len

    for i in range(0, query_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        abs_pos = query_start + offset
        mask = (offset < query_len) & (abs_pos >= query_slice_start) & (abs_pos < query_slice_stop)
        out_pos = abs_pos - query_slice_start

        # cu_seq_len_ks: row start in the gathered K buffer.
        tl.store(cu_compressed_seq_len_ks_ptr + out_pos, row_start, mask=mask)

        # cu_seq_len_ke: row start + per-token context length. Under DCP the
        # global per-token length is sharded across ranks.
        global_ctx = start_pos + 1 + offset
        len_per_token = global_ctx // COMPRESS_RATIO
        if DCP_WORLD > 1:
            # Per-rank local context length under interleave-aware DCP, matching
            # get_dcp_local_seq_lens. K == 1 reduces to (len + world-1-rank)//world.
            base = (len_per_token // DCP_INTERLEAVE // DCP_WORLD) * DCP_INTERLEAVE
            remainder = len_per_token - base * DCP_WORLD
            remainder = tl.minimum(tl.maximum(remainder - DCP_RANK * DCP_INTERLEAVE, 0), DCP_INTERLEAVE)
            len_per_token = base + remainder
        tl.store(
            cu_compressed_seq_len_ke_ptr + out_pos,
            row_start + len_per_token,
            mask=mask,
        )

    # Compute token_to_seq
    for i in range(0, compressed_seq_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < compressed_seq_len
        tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)
