# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forward_mha implementation and metadata builder for sparse MLA backends."""

from shutil import which
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

import numpy as np
import torch

from aphrodite.distributed import get_dcp_group
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.attention.mla_attention import (
    MLACommonImpl,
    MLACommonPrefillMetadata,
    build_mla_chunked_context_metadata,
    get_mla_dims,
)
from aphrodite.platforms import current_platform
from aphrodite.utils.flashinfer import has_flashinfer
from aphrodite.utils.torch_utils import np_to_pinned_tensor
from aphrodite.v1.attention.backend import (
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from aphrodite.v1.attention.backends.utils import split_decodes_and_prefills

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig
    from aphrodite.v1.attention.backend import CommonAttentionMetadata
    from aphrodite.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)


class SparseMLACommonMetadataBuilder(AttentionMetadataBuilder[T]):
    metadata_cls: type[T]
    require_uniform_decodes: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        aphrodite_config: "AphroditeConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, aphrodite_config, device)
        self.aphrodite_config = aphrodite_config
        self.device = device
        self.model_config = aphrodite_config.model_config
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens: int = aphrodite_config.model_config.hf_config.index_topk
        self.req_id_per_token_buffer = torch.empty(
            (aphrodite_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        parallel_config = aphrodite_config.parallel_config
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_local_block_size = self.cp_kv_cache_interleave_size
        self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size

        self.chunked_prefill_workspace_size = self.determine_chunked_prefill_workspace_size(aphrodite_config)
        workspace_head_size = self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
        workspace_rows = self.chunked_prefill_workspace_size
        if self.dcp_world_size > 1:
            assert self.chunked_prefill_workspace_size % self.dcp_world_size == 0
            workspace_rows += self.chunked_prefill_workspace_size // self.dcp_world_size
        self.chunked_prefill_workspace = torch.empty(
            (workspace_rows, workspace_head_size),
            dtype=self.model_config.dtype,
            device=device,
        )
        layer_prefill_backend = aphrodite_config.compilation_config.static_forward_context[
            layer_names[0]
        ].prefill_backend
        self._prefill_backend = layer_prefill_backend.clone() if layer_prefill_backend is not None else None

    @staticmethod
    def determine_chunked_prefill_workspace_size(aphrodite_config: "AphroditeConfig") -> int:
        scheduler_config = aphrodite_config.scheduler_config
        cache_config = aphrodite_config.cache_config
        model_config = aphrodite_config.model_config
        topk_tokens = model_config.hf_config.index_topk

        workspace_size = min(
            max(
                8 * model_config.max_model_len,
                4 * scheduler_config.max_num_seqs * cache_config.block_size,
            ),
            64 * 1024,
            scheduler_config.max_num_seqs * topk_tokens,
        )
        return max(workspace_size, scheduler_config.max_num_seqs * cache_config.block_size)

    def _build_req_id_per_token(self, common_attn_metadata: "CommonAttentionMetadata") -> torch.Tensor:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths)
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            np_to_pinned_tensor(req_id_per_token), non_blocking=True
        )
        return self.req_id_per_token_buffer[:num_tokens]

    def _build_chunked_context_fields(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
        num_decodes: int,
        num_prefills: int,
        prefill_query_lens_cpu: torch.Tensor | None,
    ) -> "MLACommonPrefillMetadata.ChunkedContextMetadata | None":
        if num_prefills == 0 or prefill_query_lens_cpu is None:
            return None

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        context_lens_cpu = seq_lens_cpu[num_decodes : num_decodes + num_prefills] - prefill_query_lens_cpu
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_query_start_loc_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]

        return build_mla_chunked_context_metadata(
            context_lens_cpu=context_lens_cpu,
            prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
            num_prefills=num_prefills,
            chunked_prefill_workspace=self.chunked_prefill_workspace,
            chunked_prefill_workspace_size=self.chunked_prefill_workspace_size,
            block_size=self.kv_cache_spec.block_size,
            align_chunk_to_block=current_platform.is_cuda(),
            device=self.device,
            dcp_world_size=self.dcp_world_size,
            dcp_local_block_size=self.dcp_local_block_size,
            dcp_virtual_block_size=self.dcp_virtual_block_size,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> T:
        req_id_per_token = self._build_req_id_per_token(common_attn_metadata)

        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
            require_uniform=self.require_uniform_decodes,
        )
        prefill_query_start_loc, prefill_max_query_len, prefill_query_lens_cpu = self._build_prefill_fields(
            common_attn_metadata, num_decodes, num_prefills
        )

        prefill_max_seq_len = 0
        prefill: MLACommonPrefillMetadata | None = None
        if num_prefills > 0 and self._prefill_backend is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_max_seq_len = int(seq_lens_cpu[num_decodes : num_decodes + num_prefills].max().item())
            prefill = MLACommonPrefillMetadata(
                block_table=common_attn_metadata.block_table_tensor[num_decodes:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=prefill_max_query_len,
                chunked_context=self._build_chunked_context_fields(
                    common_attn_metadata,
                    num_decodes,
                    num_prefills,
                    prefill_query_lens_cpu,
                ),
                q_data_type=self.model_config.dtype,
                output_dtype=self.model_config.dtype,
                prefill_backend=self._prefill_backend,
            )
            self._prefill_backend.prepare_metadata(prefill)

        return self.metadata_cls(  # type: ignore[call-arg]
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            prefill_max_seq_len=prefill_max_seq_len,
            prefill=prefill,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        )

    @staticmethod
    def _build_prefill_fields(
        common_attn_metadata: "CommonAttentionMetadata",
        num_decodes: int,
        num_prefills: int,
    ) -> tuple[torch.Tensor | None, int, torch.Tensor | None]:
        if num_prefills == 0:
            return None, 0, None

        offset = common_attn_metadata.query_start_loc[num_decodes]
        prefill_query_start_loc = common_attn_metadata.query_start_loc[num_decodes:] - offset

        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_qsl_cpu = qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]
        prefill_query_lens = prefill_qsl_cpu[1:] - prefill_qsl_cpu[:-1]
        prefill_max_query_len = int(prefill_query_lens.max().item())

        return prefill_query_start_loc, prefill_max_query_len, prefill_query_lens


class SparseMLACommonImpl(MLACommonImpl[Any], Generic[T]):
    """Sparse MLA base with shared dense-MHA prefill and sparse top-k MQA decode."""

    is_sparse = True

    def __init__(
        self,
        *args: Any,
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["indexer"] = indexer
        super().__init__(*args, **kwargs)
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer  # type: ignore[attr-defined]
        )
        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and which("ninja") is not None
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )
