# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 Full/Reindex/Reuse topology with native b12x compute only."""

import math
from dataclasses import replace
from typing import cast

import regex as re
import torch
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention import dsa_indexer
from b12x.attention.compressed_sparse_mla import rotary, weight_scale
from b12x.gemm import wo_projection
from torch import nn

from aphrodite.compilation.breakable_cudagraph import eager_break_during_capture
from aphrodite.distributed import get_tensor_model_parallel_world_size, get_tp_group
from aphrodite.forward_context import get_forward_context
from aphrodite.model_executor.layers.attention_layer_base import AttentionLayerBase
from aphrodite.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from aphrodite.models.deepseek_v41.common.rope import build_deepseek_v4_rope
from aphrodite.triton_utils import tl, triton
from aphrodite.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    register_b12x_unit_provider,
    set_b12x_preparation_provider,
)
from aphrodite.utils.torch_utils import current_stream
from aphrodite.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
from aphrodite.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

from .b12x_layers import (
    B12xFP8LinearMethod,
    B12xLinearMethod,
    B12xRMSNorm,
    block32_linear_plan,
    block32_linear_prequantized,
)
from .ced import CED_WINDOW, ced_decoder_start
from .compressor import DeepseekCompressor
from .qkv_norm import fused_qkv_norm_quant, prepare_qkv_norm_quant
from .sparse_mla import (
    SWA_BLOCK_SIZE,
    DeepseekV41B12xBackend,
    DeepseekV41B12xMetadata,
    _chunk,
)


def _source(prefix, layer):
    return re.sub(r"(layers\.)\d+", rf"\g<1>{layer}", prefix)


def _native_linear(layer):
    method = layer.quant_method
    if isinstance(method, (B12xFP8LinearMethod, B12xLinearMethod)):
        return
    if type(method) is UnquantizedLinearMethod:
        layer.quant_method = B12xLinearMethod()
        return
    raise ValueError("V4.1 attention only supports native b12x linear methods")


def _scratch(plan):
    return current_workspace_manager().get_simultaneous(*((s.shape, s.dtype) for s in plan.scratch_specs()))


def _rotated(x, positions, cos_sin_cache, **kwargs):
    # This result can span nested GEMMs, which reuse the workspace arena.
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rotary.rotate(x, positions, cos_sin_cache, out=out, **kwargs)
    return out


class _AttentionHelpers:
    """Loaded-resource owner, independent of unpublished attention caches."""

    def __init__(self, attention):
        self.attention = attention

    def get_b12x_preparation_units(self, layer: object, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        from b12x.preparation import FrozenMapping, MemoryRequirements, PreparedCall

        attn = self.attention
        table = attn.rotary_emb.cos_sin_cache
        device = table.device
        attn._declare_attention(device)
        roles = [("kv", 1, 512, 1), ("q", attn.n_local_heads, 512, 1)]
        if attn.indexer is not None:
            roles.append(("index_query", attn.indexer.heads, 128, 1))
        if attn.compressor is not None:
            roles.extend(
                (
                    ("index_key", 1, 128, attn.compress_ratio),
                    ("latent", 1, 512, attn.compress_ratio),
                )
            )
        plans: dict[str, object] = {}
        requests = []
        for role, heads, dim, ratio in roles:
            query = rotary.Query(
                max_rows=attn.capacity,
                heads=heads,
                dim=dim,
                ratio=ratio,
                cos_sin_dtype=str(table.dtype).removeprefix("torch."),
            )
            declaration = rotary.plan(query, device=device)
            if role == "kv":
                native_memory = declaration._memory_requirements

                def memory(config, detected, native_memory=native_memory):
                    return MemoryRequirements.sequential(
                        (
                            native_memory(config, detected),
                            attn._staging_memory(),
                        )
                    )

                declaration = replace(
                    declaration,
                    _memory_requirements=memory,
                    invocation=FrozenMapping(
                        {
                            "aphrodite_staging": tuple(
                                (name, shape, str(dtype)) for name, shape, dtype in attn._staging_specs
                            )
                        }
                    ),
                )
            plans[role] = declaration

            def prepare(state, heads=heads, dim=dim):
                attn._prepare(device)
                source = torch.ones((1, heads, dim), dtype=torch.bfloat16, device=device)
                output = torch.empty_like(source)
                positions = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(run=lambda: state.run(source, positions, table, out=output))

            requests.append(
                declaration.request(
                    name=f"{attn.prefix}.helper.{role}",
                    prepare_call=prepare,
                )
            )
        if attn.indexer is not None:
            declaration = weight_scale.plan(
                weight_scale.Query(max_elements=attn.capacity * attn.indexer.heads),
                device=device,
            )
            plans["index_weights"] = declaration

            def prepare_weights(state):
                source = torch.ones((1, attn.indexer.heads), dtype=torch.bfloat16, device=device)
                output = torch.empty_like(source)
                return PreparedCall(run=lambda: state.run(source, out=output))

            requests.append(
                declaration.request(
                    name=f"{attn.prefix}.helper.index_weights",
                    prepare_call=prepare_weights,
                )
            )
        cache_roles = [("swa_cache_write", attn.swa_cache_layer.block_size, "swa")]
        if attn.compressor is not None:
            cache_roles.append(("indexed_cache_write", attn._main_page, "indexed"))
        for role, page_size, kind in cache_roles:
            declaration = mla.plan_cache_writer(
                mla.CacheWriterQuery(
                    max_rows=attn.capacity,
                    page_size=page_size,
                    cache_kind=kind,
                    slot_dtype="int64",
                ),
                device=device,
            )
            plans[role] = declaration

            def prepare_writer(state, page_size=page_size, kind=kind):
                source = torch.ones((1, 512), dtype=torch.bfloat16, device=device)
                cache = torch.empty(
                    (
                        1,
                        mla.page_nbytes(page_size, cache_kind=kind, cache_format="deepseek_v41"),
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                slots = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(run=lambda: state.run(source, cache, slots))

            requests.append(
                declaration.request(
                    name=f"{attn.prefix}.helper.{role}",
                    prepare_call=prepare_writer,
                )
            )
        attn._helper_plans = plans
        return (
            B12xPreparationUnit(
                name="V41AttentionHelpers",
                key=(attn.prefix, attn.capacity),
                requests=tuple(requests),
                stage="weights",
            ),
            attn._wo_preparation_unit(workload),
        )


@triton.jit(do_not_specialize=["offset", "stride", "width"])
def _pages(Reqs, Table, Out, offset, stride, width, WIDTH: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    req = tl.load(Reqs + offset + row)
    page = tl.load(Table + req.to(tl.int64) * stride + col, (req >= 0) & (col < width), other=-1)
    # Sonar reserves physical block zero; b12x page tables use -1 for holes.
    page = tl.where(page > 0, page, -1)
    tl.store(Out + row * WIDTH + col, page, col < WIDTH)


class _Cache(nn.Module, AttentionLayerBase):
    def __init__(self, config, prefix, *, kind, ratio=1, window=0, draft=False):
        super().__init__()
        self.prefix, self.kind, self.ratio, self.window = prefix, kind, ratio, window
        self.draft = draft
        self.block_size = SWA_BLOCK_SIZE if kind == "swa" else None
        self.kv_cache = torch.tensor([])
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, cache):
        # Keep allocator page stride: padding belongs to the allocator, not the ABI.
        self.kv_cache = cache.view(cache.shape[0], -1)

    def get_kv_cache_spec(self, config):
        block_size = self.block_size
        if self.kind != "swa":
            block_size = (
                self.kv_cache.shape[1] // 68 * self.ratio if self.kv_cache.numel() else config.cache_config.block_size
            )
        common = dict(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512 if self.kind == "swa" else 68,
            state_content_bytes=528 if self.kind == "swa" else 68,
            dtype=torch.uint8,
            tokens_per_state=self.ratio,
            cache_dtype_str="b12x_dsv41",
            alignment=None,
            dcp_sharded=False,
            max_tp_shards=1,
        )
        if self.kind == "swa":
            boundary = ced_decoder_start(config.model_config.hf_config)
            layer_id = int(re.search(r"layers\.(\d+)", self.prefix).group(1))
            bounded_replay = self.draft or (boundary is not None and layer_id >= boundary)
            return SlidingWindowMLASpec(
                **common,
                sliding_window=self.window,
                extra_retained_tokens=int(self.draft),
                bounded_replay=bounded_replay,
            )
        return MLAAttentionSpec(**common)

    def get_attn_backend(self):
        return DeepseekV41B12xBackend

    def forward(self):
        raise RuntimeError("V4.1 caches are consumed by their owning attention layer")


class _WOProjectionWeightMethod(LinearMethodBase):
    """Keep checkpoint FP8 tensors unmodified for the fused WO projection."""

    def __init__(self, original):
        if not isinstance(original, B12xFP8LinearMethod):
            raise ValueError("V4.1 output projection requires block32 FP8 weights")
        self.original = original

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("V4.1 output projection requires FP8 checkpoint weights")
        if not hasattr(layer, "weight_scale_inv"):
            raise ValueError("V4.1 output projection requires checkpoint scales")
        if layer.weight_scale_inv.dtype != torch.float8_e8m0fnu:
            raise ValueError("V4.1 output projection requires UE8M0 checkpoint scales")

    def apply(self, layer, x, bias=None):
        raise RuntimeError("V4.1 WO weights are consumed only by the fused projection")


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config, prefix, *, owns_k, k_cache):
        super().__init__()
        hf = config.model_config.hf_config
        self.k_cache = k_cache
        # Replicating the small indexer avoids reducing a rows-by-context score
        # matrix across TP ranks. Every rank selects from all index heads.
        self.heads = hf.index_n_heads
        self.wq_b = ReplicatedLinear(
            hf.q_lora_rank,
            hf.index_n_heads * 128,
            bias=False,
            return_bias=False,
            quant_config=config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        _native_linear(self.wq_b)
        self.weights_proj = ReplicatedLinear(
            hf.hidden_size,
            hf.index_n_heads,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.weights_proj.quant_method = B12xLinearMethod()
        if owns_k:
            self.wk = ReplicatedLinear(
                512,
                128,
                bias=False,
                return_bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
            self.wk.quant_method = B12xLinearMethod()
            self.k_norm = B12xRMSNorm(128, hf.rms_norm_eps)


@torch.library.custom_op("aphrodite::dsv41_b12x_attention", mutates_args=())
def _attention(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
    global_kv_ready: torch.Tensor | None,
) -> torch.Tensor:
    context = get_forward_context()
    layer = context.no_compile_layers[prefix]
    # WO projection and TP reduction own this result; it cannot alias the
    # borrowed attention scratch or the operation's input tensors.
    return layer._forward(positions, hidden)


@_attention.register_fake
def _attention_fake(hidden, positions, prefix, global_kv_ready):
    return torch.empty_like(hidden, memory_format=torch.contiguous_format)


@torch.library.custom_op("aphrodite::dsv41_b12x_prepare_global_kv", mutates_args=())
def _prepare_global_kv(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
) -> torch.Tensor:
    context = get_forward_context()
    context.no_compile_layers[prefix]._prepare_global_kv(positions, hidden)
    # A small explicit data dependency orders the following opaque attention
    # call without asking functionalization to clone the aliased KV pool.
    return torch.empty(1, dtype=torch.uint8, device=hidden.device)


@_prepare_global_kv.register_fake
def _prepare_global_kv_fake(hidden, positions, prefix):
    return torch.empty(1, dtype=torch.uint8, device=hidden.device)


class DeepseekV41B12xAttention(nn.Module, AttentionLayerBase):
    backend_cls = DeepseekV41B12xBackend
    # Index scores need bounded row scratch; attention handles the whole batch.
    INDEX_CHUNK = 256
    DECODE_CHUNK = 64

    def __init__(
        self,
        aphrodite_config,
        prefix,
        topk_indices_buffer,
    ):
        super().__init__()
        self.config = aphrodite_config
        hf = aphrodite_config.model_config.hf_config
        self.prefix = prefix
        layer_match = re.search(r"layers\.(\d+)", prefix)
        if layer_match is None:
            raise ValueError("V4.1 attention prefix must contain a layer index")
        self.layer_id = int(layer_match.group(1))
        self.hidden_size, self.head_dim = hf.hidden_size, hf.head_dim
        self.rope_head_dim = hf.qk_rope_head_dim
        tp = get_tensor_model_parallel_world_size()
        if (
            hf.num_attention_heads <= 0
            or hf.o_groups <= 0
            or hf.num_attention_heads % tp
            or hf.o_groups % tp
            or hf.num_attention_heads % hf.o_groups
        ):
            raise ValueError("V4.1 TP must evenly shard complete attention output groups")
        self.n_local_heads = hf.num_attention_heads // tp
        self.n_local_groups = hf.o_groups // tp
        self.o_lora_rank = hf.o_lora_rank
        self.q_lora_rank, self.window_size = hf.q_lora_rank, hf.sliding_window
        spec = aphrodite_config.speculative_config
        self.is_draft = self.layer_id >= hf.num_hidden_layers and spec is not None and spec.use_dspark()
        boundary = ced_decoder_start(hf)
        self.is_ced_decoder = not self.is_draft and boundary is not None and self.layer_id >= boundary
        self.swa_width = self.window_size
        if self.is_draft:
            from aphrodite.v1.attention.backends.mla.compressor_utils import (
                get_dspark_swa_index_width,
            )

            self.swa_width = get_dspark_swa_index_width(self.window_size, spec.num_speculative_tokens)
        self.capacity = aphrodite_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = aphrodite_config.model_config.max_model_len
        self.compress_ratio = hf.compress_ratios[self.layer_id] if self.layer_id < len(hf.compress_ratios) else 0
        self.is_kv_source = self.layer_id in hf.kv_source_layer_ids
        self.is_index_source = self.layer_id in hf.index_source_layer_ids
        self.kv_source_layer_id = (
            max(
                (s for s in hf.kv_source_layer_ids if s <= self.layer_id),
                default=None,
            )
            if self.compress_ratio
            else None
        )
        self.index_source_layer_id = (
            max(
                (s for s in hf.index_source_layer_ids if s <= self.layer_id),
                default=None,
            )
            if self.compress_ratio
            else None
        )
        self.candidate_source_layer = hf.candidate_source_layer_id
        if self.head_dim != 512 or self.rope_head_dim != 64 or self.compress_ratio not in (0, 1, 2):
            raise ValueError("unsupported V4.1 native attention geometry")
        if self.compress_ratio:
            sources = (self.kv_source_layer_id, self.index_source_layer_id)
            if any(
                source is None
                or source < 0
                or source >= len(hf.compress_ratios)
                or hf.compress_ratios[source] != self.compress_ratio
                for source in sources
            ):
                raise ValueError(
                    "V4.1 compressed layers require preceding KV/index sources with the same compression ratio"
                )
            if self.is_kv_source and not self.is_index_source:
                raise ValueError("V4.1 native KV sources must also own index keys")
        elif self.is_kv_source or self.is_index_source:
            raise ValueError("V4.1 SWA-only layers cannot own compressed caches")
        if (
            hf.index_head_dim != 128
            or hf.index_n_heads <= 0
            or hf.index_n_heads > 32
            or 32 % hf.index_n_heads
            or hf.index_topk != 512
            or hf.candidate_topk_blocks != 2048
            or hf.candidate_block_size != 8
            or self.hidden_size % 32
            or self.o_lora_rank % 32
        ):
            raise ValueError("unsupported V4.1 native indexer or WO geometry")
        if aphrodite_config.cache_config.block_size % 32:
            raise ValueError("V4.1 native cache blocks must be a multiple of 32")
        if aphrodite_config.parallel_config.use_ubatching:
            raise ValueError("V4.1 native attention does not support DBO or ubatching")
        if aphrodite_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("V4.1 native cross-layer cache ownership does not support PP")
        if (
            aphrodite_config.parallel_config.decode_context_parallel_size != 1
            or aphrodite_config.parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError("V4.1 TP shards heads, not context; context parallel is unsupported")
        self._context = aphrodite_config.compilation_config.static_forward_context
        if prefix in self._context:
            raise ValueError(f"Duplicate attention layer {prefix}")
        self._context[prefix] = self
        self.kv_cache = torch.tensor([])
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            hf.hidden_size,
            [hf.q_lora_rank, 512],
            bias=False,
            quant_config=aphrodite_config.quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_wqa_wkv",
        )
        self.q_norm = B12xRMSNorm(hf.q_lora_rank, hf.rms_norm_eps)
        self.kv_norm = B12xRMSNorm(512, hf.rms_norm_eps)
        self._qkv_norm_quant = False
        self.wq_b = ColumnParallelLinear(
            hf.q_lora_rank,
            hf.num_attention_heads * 512,
            bias=False,
            return_bias=False,
            quant_config=aphrodite_config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = ColumnParallelLinear(
            hf.num_attention_heads * 512 // hf.o_groups,
            hf.o_groups * hf.o_lora_rank,
            bias=False,
            return_bias=False,
            quant_config=aphrodite_config.quant_config,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.quant_method = _WOProjectionWeightMethod(self.wo_a.quant_method)
        self.wo_b = RowParallelLinear(
            hf.o_groups * hf.o_lora_rank,
            hf.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=aphrodite_config.quant_config,
            prefix=f"{prefix}.wo_b",
            reduce_results=False,
        )
        self.wo_b.quant_method = _WOProjectionWeightMethod(self.wo_b.quant_method)
        self._wo_projection_weights = None
        self._wo_plans = {}
        for linear in (self.fused_wqa_wkv, self.wq_b):
            _native_linear(linear)
        self.rotary_emb = build_deepseek_v4_rope(
            hf,
            head_dim=512,
            rope_head_dim=64,
            max_position_embeddings=hf.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.swa_cache_layer = _Cache(
            aphrodite_config,
            f"{prefix}.swa_cache",
            kind="swa",
            window=self.window_size,
            draft=self.is_draft,
        )
        self.compressor = (
            DeepseekCompressor(
                aphrodite_config,
                self.compress_ratio,
                hf.hidden_size,
                512,
                prefix=f"{prefix}.compressor",
            )
            if self.is_kv_source
            else None
        )
        self.indexer = None
        if self.is_index_source:
            if self.is_kv_source:
                k_cache = _Cache(
                    aphrodite_config,
                    f"{prefix}.indexer.k_cache",
                    kind="index",
                    ratio=self.compress_ratio,
                )
            else:
                k_cache = self._context[f"{_source(prefix, self.kv_source_layer_id)}.indexer.k_cache"]
            self.indexer = DeepseekV4Indexer(
                aphrodite_config,
                f"{prefix}.indexer",
                owns_k=self.is_kv_source,
                k_cache=k_cache,
            )
        self.topk_indices_buffer = topk_indices_buffer
        self._index_plans: dict[tuple[str, int], object] = {}
        self._attention_plans: dict[str, object] = {}
        self._helper_plans: dict[str, object] = {}
        self._helpers = _AttentionHelpers(self)
        register_b12x_unit_provider(self._helpers, owner=self)
        set_b12x_preparation_provider(self, self)
        self._ready = False

    def get_attn_backend(self):
        return self.backend_cls

    def get_kv_cache_spec(self, config):
        if not self.is_kv_source:
            return None
        block_size = (
            self.kv_cache.shape[1] // 288 * self.compress_ratio
            if self.compress_ratio and self.kv_cache.numel()
            else config.cache_config.block_size
        )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512,
            state_content_bytes=288,
            dtype=torch.uint8,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str="b12x_dsv41",
            alignment=None,
            dcp_sharded=False,
            max_tp_shards=1,
        )

    def bind_kv_cache(self, cache):
        self.kv_cache = cache.view(cache.shape[0], -1)

    def _owner(self):
        return self._context[_source(self.prefix, self.kv_source_layer_id)]

    def _declare_attention(self, device):
        """Describe native code and simultaneous metadata without allocating."""
        from b12x._lib.scratch import ScratchBufferSpec
        from b12x.preparation import MemoryRequirements

        def descriptor(tensor, *, page_size, kind):
            if tensor is None or tensor.numel() == 0:
                width = mla.page_nbytes(
                    page_size,
                    cache_kind=kind,
                    cache_format="deepseek_v41",
                )
                return {
                    "shape": (1, width),
                    "stride": (width, 1),
                    "alignment": 16,
                    "dtype": "uint8",
                }
            return {
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
                "alignment": min(16, tensor.data_ptr() & -tensor.data_ptr()),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }

        ratio = max(self.compress_ratio, 1)
        main_cache = self._owner().kv_cache if self.compress_ratio else None
        self._main_page = (
            main_cache.shape[1] // 288
            if main_cache is not None and main_cache.numel()
            else self.config.cache_config.block_size // ratio
        )
        self._main_width = triton.cdiv(self.max_model_len, self._main_page * ratio)
        index_cache = self.indexer.k_cache.kv_cache if self.indexer is not None else None
        self._index_page = (
            index_cache.shape[1] // 68 if index_cache is not None and index_cache.numel() else self._main_page
        )
        self._index_width = triton.cdiv(self.max_model_len, self._index_page * ratio)
        spec = self.config.speculative_config
        # Parallel drafting can admit two draft spans during verifier profiling.
        query_width = 1 + (2 if spec.parallel_drafting else 1) * spec.num_speculative_tokens if spec is not None else 1
        decode_rows = self.config.scheduler_config.max_num_seqs * query_width
        # Graph buffers include padding beyond the live decode-token bound.
        decode_rows = max(
            decode_rows,
            self.config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self._attention_workspace_specs = {}
        self._attention_declarations = {}
        for mode, capacity in (
            ("decode", min(self.capacity, decode_rows)),
            ("extend", self.capacity),
        ):
            caps = mla.Caps(
                device=device,
                num_q_heads=self.n_local_heads,
                max_q_rows=capacity,
                max_width=self.swa_width + (512 if self.compress_ratio else 0),
                swa_width=self.swa_width,
                indexed_width=512 if self.compress_ratio else 0,
                swa_page_size=self.swa_cache_layer.block_size,
                indexed_page_size=self._main_page,
                max_page_table_width=self._main_width,
                mode=mode,
                # DSpark's reservation can exceed the generic decode cutoff.
                # Keep it on the decode split contract; prefill has a separate plan.
                decode_row_capacity=capacity if mode == "decode" else None,
                cache_format="deepseek_v41",
                use_cuda_graph=True,
            )
            swa_cache = getattr(self.swa_cache_layer, "kv_cache", None)
            indexed_cache = getattr(self._owner(), "kv_cache", None) if self.compress_ratio else None
            invocation = mla.invocation_from_descriptors(
                q={
                    "shape": (capacity, self.n_local_heads, 512),
                    "stride": (self.n_local_heads * 512, 512, 1),
                    "alignment": 16,
                    "dtype": "bfloat16",
                },
                swa_cache=descriptor(swa_cache, page_size=self.swa_cache_layer.block_size, kind="swa"),
                indexed_cache=(
                    descriptor(indexed_cache, page_size=self._main_page, kind="indexed")
                    if self.compress_ratio
                    else None
                ),
                attn_sink_present=True,
                output_mode="provided",
            )
            declaration = mla.plan(caps, invocation=invocation)
            # Pin BF16 arithmetic without changing the plan's split/single-pass choice.
            default = declaration.contract.default_config(declaration.query, None)
            declaration = mla.plan(
                caps,
                invocation=invocation,
                override=replace(default, v41_compute_mode="bf16", v41_heads_per_block=16),
            )
            metadata_specs: tuple[tuple[tuple[int, ...], torch.dtype], ...] = (
                ((capacity, self.swa_width), torch.int32),
                ((capacity,), torch.int32),
                ((capacity,), torch.int32),
            )
            if self.compress_ratio:
                metadata_specs += (((capacity, self._main_width), torch.int32),)
            self._attention_workspace_specs[mode] = metadata_specs

            # These metadata views coexist with native scratch in one arena.
            # Only integration-owned storage is added; launch policy stays native.
            def memory(
                config,
                detected,
                native=declaration._memory_requirements,
                metadata_specs=metadata_specs,
                mode=mode,
            ):
                requirement = native(config, detected)
                metadata = tuple(
                    ScratchBufferSpec(
                        name=f"{mode}_metadata_{index}",
                        shape=shape,
                        dtype=dtype,
                        device=torch.device("cuda", detected.ordinal),
                    )
                    for index, (shape, dtype) in enumerate(metadata_specs)
                )
                return MemoryRequirements(
                    scratch=(*metadata, *requirement.scratch),
                    persistent=requirement.persistent,
                )

            self._attention_declarations[mode] = replace(
                declaration,
                _memory_requirements=memory,
            )
        specs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        if self.indexer is not None:
            specs.extend(
                (
                    (
                        "_index_pages",
                        (self.INDEX_CHUNK, self._index_width),
                        torch.int32,
                    ),
                    ("_active", (1,), torch.int32),
                )
            )
            if self.layer_id == self.candidate_source_layer:
                specs.extend(
                    (
                        ("_candidates", (self.capacity, 16384), torch.int32),
                        ("_candidate_lens", (self.capacity,), torch.int32),
                    )
                )
        self._short_index_shape = None
        if (
            self.indexer is not None
            and not self.is_ced_decoder
            and self.capacity > self.INDEX_CHUNK
            and self.layer_id < self.candidate_source_layer
        ):
            short_rows = min(1024, self.capacity)
            short_width = min(self._index_width, triton.cdiv(16384, self._index_page))
            self._short_index_shape = (short_rows, short_width)
            specs.append(("_short_index_pages", self._short_index_shape, torch.int32))
        self._staging_specs = tuple(specs)

    def _staging_memory(self):
        from b12x.preparation import MemoryRequirements, PersistentMemory

        required = resident = 0
        for name, shape, dtype in self._staging_specs:
            nbytes = math.prod(shape) * dtype.itemsize
            required += nbytes
            tensor = getattr(self, name, None)
            if tensor is not None and tuple(tensor.shape) == shape and tensor.dtype == dtype:
                resident += nbytes
        return MemoryRequirements(
            persistent=(
                PersistentMemory(
                    key=(self, "attention_metadata"),
                    required_nbytes=required,
                    resident_nbytes=resident,
                ),
            )
        )

    def _prepare(self, device):
        """Materialize admitted integration-owned metadata, never launch choices."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 metadata must be prepared before graph capture")
        if not hasattr(self, "_staging_specs"):
            self._declare_attention(device)
        for name, shape, dtype in self._staging_specs:
            tensor = getattr(self, name, None)
            if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                setattr(self, name, torch.empty(shape, dtype=dtype, device=device))
        if self.indexer is not None:
            self._active.fill_(self._index_width * self._index_page)
        self._ready = True

    def _preparation_token_counts(self, workload):
        counts = set(workload.token_counts)
        if self.is_ced_decoder:
            compact_max = min(self.capacity, workload.max_seqs * CED_WINDOW)
            counts.add(compact_max)
            counts.update(range(CED_WINDOW, compact_max + 1, CED_WINDOW))
        return tuple(sorted(counts))

    def _index_request_name(self, mode: str, rows: int) -> str:
        return f"{self.prefix}.mxfp4_index.{mode}.m{rows}"

    def get_b12x_preparation_units(self, layer: object, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        device = self.rotary_emb.cos_sin_cache.device
        self._declare_attention(device)
        index_declarations = {}
        token_counts = self._preparation_token_counts(workload)
        if self.indexer is not None:
            regimes = [
                ("decode", self.DECODE_CHUNK, self._index_width),
                ("prefill", self.INDEX_CHUNK, self._index_width),
            ]
            if self._short_index_shape is not None:
                regimes.append(("prefill_short", *self._short_index_shape))
            for mode, chunk, width in regimes:
                counts: set[int] = set()
                counts.update(min(rows, chunk) for rows in token_counts)
                counts.update(rows % chunk for rows in token_counts if rows % chunk)
                for rows in sorted(counts):
                    index_declarations[mode, rows] = self._declare_index_plan(mode, rows)

        caches = [getattr(self.swa_cache_layer, "kv_cache", None)]
        if self.compress_ratio:
            caches.append(getattr(self._owner(), "kv_cache", None))
        if self.indexer is not None:
            caches.append(getattr(self.indexer.k_cache, "kv_cache", None))
        if any(cache is None or cache.numel() == 0 for cache in caches):
            return ()
        self._index_plans = dict(index_declarations)
        requests = [
            declaration.request(
                name=self._index_request_name(mode, rows),
                prepare_call=self._index_call(mode, rows),
                benchmark_call=self._index_call(mode, rows, benchmark=True),
            )
            for (mode, rows), declaration in index_declarations.items()
        ]
        self._attention_plans = dict(self._attention_declarations)
        for mode, declaration in self._attention_declarations.items():
            capacity = declaration.query.query_rows
            requests.append(
                declaration.request(
                    name=f"{self.prefix}.mla.{mode}.m{capacity}",
                    prepare_call=self._attention_call(mode, capacity),
                    benchmark_call=self._attention_call(mode, capacity),
                )
            )
        return (
            B12xPreparationUnit(
                name="V41Attention",
                key=(self.prefix, token_counts),
                requests=tuple(requests),
                stage="state",
            ),
        )

    def _attention_call(self, mode: str, rows: int):
        def prepare(state):
            from b12x.preparation import PreparedCall

            device = self.swa_cache_layer.kv_cache.device
            source = torch.randn((rows, self.n_local_heads, 512), dtype=torch.bfloat16, device=device)
            q = torch.empty_like(source)
            out = torch.empty_like(source)
            # Trial scratch belongs to the trial; the serving workspace is
            # never drawn from while candidates are timed.
            scratch = [
                torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in state.scratch_plan.scratch_specs()
            ]
            snapshots = []
            initializers = []

            def cache_inputs(cache, page_size, width, kind):
                pages = min(
                    cache.shape[0],
                    self._main_width if kind == "indexed" else triton.cdiv(width, page_size),
                )
                if pages < 1:
                    raise PreparationResourceUnavailableError("attention primer needs a live cache page")
                live = cache[:pages]
                snapshots.append((live, live.clone()))
                records = live.view(pages, page_size, -1)
                if kind == "swa":
                    payload = torch.randn((pages, page_size, 512), device=device).mul_(0.25)
                    initializers.append(
                        (
                            records,
                            payload.to(torch.float8_e4m3fn).view(torch.uint8),
                            512,
                            127,
                        )
                    )
                else:
                    payload = torch.randint(
                        0,
                        256,
                        (pages, page_size, 256),
                        dtype=torch.uint8,
                        device=device,
                    )
                    initializers.append((records, payload, 256, 56))
                count = min(width, pages * page_size)
                indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
                indices[:, :count] = torch.arange(count, dtype=torch.int32, device=device)
                lengths = torch.full((rows,), count, dtype=torch.int32, device=device)
                return indices, lengths, pages

            swa_indices, swa_lengths, _ = cache_inputs(
                self.swa_cache_layer.kv_cache,
                self.swa_cache_layer.block_size,
                self.swa_width,
                "swa",
            )
            kwargs = {}
            if self.compress_ratio:
                indices, lengths, pages = cache_inputs(
                    self._owner().kv_cache,
                    self._main_page,
                    512,
                    "indexed",
                )
                table = torch.full((rows, self._main_width), -1, dtype=torch.int32, device=device)
                table[:, :pages] = torch.arange(pages, dtype=torch.int32, device=device)
                kwargs = dict(
                    indexed_indices=indices,
                    indexed_lengths=lengths,
                    indexed_page_table=table,
                )
            binding = state.bind_for_preparation(
                scratch=scratch,
                q=q,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                **kwargs,
            )

            def reset():
                for target, payload, width, scale in initializers:
                    target[..., :width].copy_(payload)
                    target[..., width:].fill_(scale)
                out.fill_(float("nan"))

            def restore():
                from b12x.preparation.types import _close_all

                _close_all((lambda target=target, saved=saved: target.copy_(saved)) for target, saved in snapshots)

            return PreparedCall(
                run=lambda: state.run(
                    binding,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    indexed_k_cache=self._owner().kv_cache if self.compress_ratio else None,
                    swa_page_size=self.swa_cache_layer.block_size,
                    indexed_page_size=self._main_page,
                    sm_scale=512**-0.5,
                    attn_sink=self.attn_sink,
                    out=out,
                    cache_format="deepseek_v41",
                ),
                output=out,
                produce=lambda: q.copy_(source),
                reset=reset,
                restore=restore,
            )

        return prepare

    def _index_call(self, mode: str, rows: int, *, benchmark: bool = False):
        def prepare(state):
            from b12x.attention.dsa_indexer.mxfp4 import score_mxfp4, select_mxfp4
            from b12x.preparation import PreparedCall

            assert self.indexer is not None
            cache = self.indexer.k_cache.kv_cache
            device, heads = cache.device, self.indexer.heads
            width = state.caps.max_page_table_width
            live_pages = min(cache.shape[0], width)
            if live_pages < 1:
                raise PreparationResourceUnavailableError("indexer primer needs a live cache page")
            live_tokens = live_pages * self._index_page
            saved_cache = cache[:live_pages].clone()
            keys = torch.randn((live_tokens, 128), dtype=torch.bfloat16, device=device)
            slots = torch.arange(live_tokens, dtype=torch.int64, device=device)
            source = torch.randn((rows, heads, 128), dtype=torch.bfloat16, device=device)
            q = torch.empty((rows, heads, 64), dtype=torch.uint8, device=device)
            scales = torch.empty((rows, heads, 4), dtype=torch.uint8, device=device)
            weights = torch.full((rows, heads), 1 / 64, dtype=torch.bfloat16, device=device)
            pages = torch.full((rows, width), -1, dtype=torch.int32, device=device)
            pages[:, :live_pages] = torch.arange(live_pages, dtype=torch.int32, device=device)
            lengths = torch.full((rows,), live_tokens, dtype=torch.int32, device=device)
            active = torch.full((1,), live_tokens, dtype=torch.int32, device=device)
            indices = torch.empty((rows, 512), dtype=torch.int32, device=device)
            kwargs = {}
            if self.layer_id == self.candidate_source_layer:
                kwargs = dict(
                    candidate_output=torch.empty((rows, 16384), dtype=torch.int32, device=device),
                    candidate_output_lengths=torch.empty((rows,), dtype=torch.int32, device=device),
                )
            elif self.layer_id > self.candidate_source_layer:
                count = min(16384, live_tokens)
                candidates = torch.full((rows, 16384), -1, dtype=torch.int32, device=device)
                candidates[:, :count] = torch.arange(count, dtype=torch.int32, device=device)
                kwargs = dict(
                    candidate_indices=candidates,
                    candidate_lengths=torch.full((rows,), count, dtype=torch.int32, device=device),
                )
            # Retained tuning trials need independent scratch; priming is serial.
            scratch = (
                [torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in state.layout.scratch_specs()]
                if benchmark
                else _scratch(state.layout)
            )
            binding = state.bind(
                scratch=scratch,
                q_mxfp4=q,
                q_scales=scales,
                query_weights=weights,
                index_k_cache=cache,
                page_table=pages,
                cache_lengths=lengths,
                active_width=active,
                output_indices=indices,
                **kwargs,
            )

            def run():
                score_mxfp4(binding, launchers=state._launchers)
                return select_mxfp4(binding, launchers=state._launchers)

            return PreparedCall(
                run=run,
                output=indices,
                produce=lambda: state.quantize_query(source, q_mxfp4=q, q_scales=scales),
                reset=lambda: state.write_index_keys(keys, index_k_cache=cache, slot_mapping=slots),
                restore=lambda: cache[:live_pages].copy_(saved_cache),
            )

        return prepare

    def _helper_plan(self, role: str):
        try:
            return self._helper_plans[role]
        except KeyError:
            raise PreparationResourceUnavailableError(
                f"{self.prefix} lacks a declared {role} attention helper plan"
            ) from None

    def _attention_plan(self, mode: str):
        try:
            return self._attention_plans[mode]
        except KeyError:
            raise PreparationResourceUnavailableError(
                f"{self.prefix} lacks a declared {mode} MLA attention plan"
            ) from None

    def _declare_index_plan(self, mode: str, rows: int):
        if mode == "decode":
            chunk, width = self.DECODE_CHUNK, self._index_width
        elif mode == "prefill":
            chunk, width = self.INDEX_CHUNK, self._index_width
        elif mode == "prefill_short" and self._short_index_shape is not None:
            chunk, width = self._short_index_shape
        else:
            raise ValueError(f"unsupported V4.1 indexer mode {mode!r}")
        if not 0 < rows <= chunk:
            raise ValueError(f"V4.1 {mode} indexer rows {rows} exceed chunk capacity {chunk}")
        assert self.indexer is not None
        return dsa_indexer.plan(
            dsa_indexer.Caps(
                device=self.rotary_emb.cos_sin_cache.device,
                num_q_heads=self.indexer.heads,
                max_q_rows=rows,
                max_page_table_width=width,
                topk=512,
                mode="decode" if mode == "decode" else "prefill",
                cache_format="mxfp4",
                page_size=self._index_page,
                max_candidates=16384 if self.layer_id > self.candidate_source_layer else 0,
                candidate_topk_blocks=2048 if self.layer_id == self.candidate_source_layer else 0,
            )
        )

    def _index_plan(self, mode: str, rows: int):
        plan = self._index_plans.get((mode, rows))
        if plan is not None:
            return plan
        if mode == "decode":
            capacity = self.DECODE_CHUNK
        elif mode == "prefill":
            capacity = self.INDEX_CHUNK
        elif mode == "prefill_short" and self._short_index_shape is not None:
            capacity = self._short_index_shape[0]
        else:
            raise ValueError(f"unsupported V4.1 indexer mode {mode!r}")
        if not 0 < rows <= capacity:
            raise ValueError(f"V4.1 {mode} indexer rows {rows} exceed chunk capacity {capacity}")
        # Only session-prepared plans can run. Keep the full chunk plan when it
        # was prepared; otherwise use the smallest prepared plan that covers rows
        # (a token budget below the chunk never declares the full chunk).
        covering = sorted(prepared for kind, prepared in self._index_plans if kind == mode and prepared >= rows)
        if covering:
            return self._index_plans[mode, capacity if capacity in covering else covering[0]]
        raise PreparationResourceUnavailableError(f"{self.prefix} lacks a prepared {mode} indexer plan for {rows} rows")

    def insert_context_kv(self, kv, positions, slot_mapping, *, rotated=False):
        if not self._ready:
            raise PreparationResourceUnavailableError("V4.1 attention metadata is not prepared")
        if not rotated:
            kv = _rotated(
                kv,
                positions,
                self.rotary_emb.cos_sin_cache,
                plan=self._helper_plan("kv"),
            )
        mla.write_cache(
            kv,
            self.swa_cache_layer.kv_cache,
            slot_mapping,
            page_size=self.swa_cache_layer.block_size,
            cache_kind="swa",
            cache_format="deepseek_v41",
            plan=self._helper_plan("swa_cache_write"),
        )

    def _query_metadata(self, metadata):
        if self.is_ced_decoder and metadata.decoder is not None:
            return metadata.decoder
        return metadata

    @eager_break_during_capture
    def _cache_context_kv(self, kv, positions, *, rotated=False):
        # Resolve live request metadata inside the eager segment. A captured
        # slot-mapping argument could refer to a previous request's cache pages.
        metadata = get_forward_context().attn_metadata
        swa = self._query_metadata(metadata[self.swa_cache_layer.prefix])
        self.insert_context_kv(kv, positions, swa.slot_mapping[: kv.shape[0]], rotated=rotated)

    def prepare_global_kv(self, positions, hidden_states):
        """Build full-row global KV before the CED boundary gathers decoder rows."""
        if self.compressor is None or hidden_states.shape[0] == 0:
            return
        return _prepare_global_kv(hidden_states, positions, self.prefix)

    @eager_break_during_capture
    def _prepare_global_kv(self, positions, hidden_states):
        if not self._ready:
            raise PreparationResourceUnavailableError("V4.1 attention metadata is not prepared")
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or self.compressor is None:
            return
        indexer = self.indexer
        if indexer is None:
            raise RuntimeError("V4.1 global KV source requires an indexer")
        rows = hidden_states.shape[0]
        # These are deliberately original metadata, never the decoder views.
        main = metadata[self._owner().prefix]
        state = metadata[self.compressor.state_cache.prefix] if self.compressor.state_cache is not None else None
        latent, slots = self.compressor(hidden_states, main, state)
        # Index K consumes the ordinary-normalized PRE-RoPE latent.
        key = indexer.k_norm(indexer.wk(latent))
        key = _rotated(
            key,
            positions,
            self.rotary_emb.cos_sin_cache,
            plan=self._helper_plan("index_key"),
        )
        index_meta = cast(DeepseekV41B12xMetadata, metadata[indexer.k_cache.prefix])
        for offset in range(0, rows, self.INDEX_CHUNK):
            end = min(offset + self.INDEX_CHUNK, rows)
            dsa_indexer.quantize_write_index_k_mxfp4(
                self._index_plan("prefill", end - offset),
                key[offset:end],
                index_k_cache=indexer.k_cache.kv_cache,
                slot_mapping=index_meta.slot_mapping[offset:end],
            )
        latent = _rotated(
            latent,
            positions,
            self.rotary_emb.cos_sin_cache,
            plan=self._helper_plan("latent"),
        )
        mla.write_cache(
            latent,
            self.kv_cache,
            slots,
            page_size=self._main_page,
            cache_kind="indexed",
            cache_format="deepseek_v41",
            plan=self._helper_plan("indexed_cache_write"),
        )

    def forward(self, positions, hidden_states, *, global_kv_ready=None):
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        return _attention(hidden_states, positions, self.prefix, global_kv_ready)

    def _forward(self, positions, hidden_states):
        if not self._ready:
            raise PreparationResourceUnavailableError("V4.1 attention metadata is not prepared")
        rows = hidden_states.shape[0]
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        norm_quant = self._qkv_norm_quant and rows <= self.DECODE_CHUNK and qr_kv.dtype == torch.bfloat16
        qr = q_values = q_scale_mma = None
        if norm_quant:
            q_plan = block32_linear_plan(self.wq_b, rows)
            q_values, q_scale_mma, kv = fused_qkv_norm_quant(
                qr_kv,
                self.q_norm.weight,
                self.kv_norm.weight,
                self.q_norm.variance_epsilon,
                q_plan.query.max_tokens,
                positions,
                self.rotary_emb.cos_sin_cache,
            )
            q = block32_linear_prequantized(self.wq_b, q_plan, q_values, q_scale_mma)
        else:
            qr, kv = qr_kv.split((self.q_lora_rank, 512), dim=-1)
            qr, kv = self.q_norm(qr), self.kv_norm(kv)
            q = self.wq_b(qr)
        q = q.view(rows, self.n_local_heads, 512)
        q = _rotated(
            q,
            positions,
            self.rotary_emb.cos_sin_cache,
            plan=self._helper_plan("q"),
        )
        output = torch.empty_like(q)
        metadata = cast(
            dict[str, DeepseekV41B12xMetadata] | None,
            get_forward_context().attn_metadata,
        )
        is_prefill = True
        if not isinstance(metadata, dict):
            # Memory profiling has no cache pages; serving does not use this branch.
            output.zero_()
        else:
            original_swa = metadata[self.swa_cache_layer.prefix]
            swa = self._query_metadata(original_swa)
            is_prefill = not swa.is_decode
            # The fused path already applied the native KV rotation.
            self._cache_context_kv(kv, positions, rotated=norm_quant)
            if swa is original_swa:
                self._prepare_global_kv(positions, hidden_states)
            index_query = None
            if self.indexer is not None:
                h = self.indexer.heads
                if norm_quant:
                    index_plan = block32_linear_plan(self.indexer.wq_b, rows)
                    iq = block32_linear_prequantized(self.indexer.wq_b, index_plan, q_values, q_scale_mma)
                else:
                    iq = self.indexer.wq_b(qr)
                iq = iq.view(rows, h, 128)
                iq = _rotated(
                    iq,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    plan=self._helper_plan("index_query"),
                )
                iq_data = torch.empty((rows, h, 64), dtype=torch.uint8, device=q.device)
                iq_scale = torch.empty((rows, h, 4), dtype=torch.uint8, device=q.device)
                index_mode = "decode" if swa.is_decode else "prefill"
                chunk_rows = self.DECODE_CHUNK if swa.is_decode else self.INDEX_CHUNK
                for offset in range(0, rows, chunk_rows):
                    end = min(offset + chunk_rows, rows)
                    dsa_indexer.quantize_q_mxfp4(
                        self._index_plan(index_mode, end - offset),
                        iq[offset:end],
                        q_mxfp4=iq_data[offset:end],
                        q_scales=iq_scale[offset:end],
                    )
                weights = self.indexer.weights_proj(hidden_states)
                iw = torch.empty_like(weights)
                weight_scale.scale_index_weights(
                    weights,
                    out=iw,
                    plan=self._helper_plan("index_weights"),
                )
                index_query = (iq_data, iq_scale, iw)
            self.forward_mqa(q, kv, positions, output, index_query=index_query)
        return self._o_proj(output, positions, is_prefill=is_prefill)

    @eager_break_during_capture
    def forward_mqa(self, q, kv, positions, output, *, index_query=None):
        metadata = get_forward_context().attn_metadata
        swa = self._query_metadata(metadata[self.swa_cache_layer.prefix])
        main = self._query_metadata(metadata[self._owner().prefix]) if self.compress_ratio else None
        mode = "decode" if swa.is_decode else "extend"
        rows = q.shape[0]
        from b12x.preparation import require_prepared

        plan = self._attention_plan(mode)
        state = require_prepared(plan, "attention.compressed_sparse_mla")
        if rows > state.query.query_rows:
            raise ValueError(f"V4.1 {mode} rows {rows} exceed prepared capacity {state.query.query_rows}")
        owner = self._context[_source(self.prefix, self.index_source_layer_id)] if self.compress_ratio else self

        # Only the score matrix needs row chunking. All selected positions
        # survive in the source-owned buffer until their reuse interval ends.
        if self.indexer is not None:
            index_mode = "decode" if mode == "decode" else "prefill"
            iq_data, iq_scale, iw = index_query
            im = self._query_metadata(metadata[self.indexer.k_cache.prefix])
            score_width = None
            if (
                mode == "extend"
                and self.layer_id <= self.candidate_source_layer
                and not torch.cuda.is_current_stream_capturing()
            ):
                score_width = min(
                    self._index_width * self._index_page,
                    max(1, triton.cdiv(im.max_seq_len, self.compress_ratio)),
                )
            chunk_rows = self.DECODE_CHUNK if mode == "decode" else self.INDEX_CHUNK
            index_pages = self._index_pages
            if (
                mode == "extend"
                and score_width is not None
                and score_width <= 16384
                and self._short_index_shape is not None
            ):
                index_mode = "prefill_short"
                chunk_rows = self._short_index_shape[0]
                index_pages = self._short_index_pages
            index_width = index_pages.shape[1]
            for offset in range(0, rows, chunk_rows):
                end = min(offset + chunk_rows, rows)
                count = end - offset
                _pages[(count, triton.cdiv(index_width, 128))](
                    im.req_id_per_token,
                    im.block_table,
                    index_pages,
                    offset,
                    im.block_table.stride(0),
                    min(im.block_table.shape[1], index_width),
                    index_width,
                    128,
                )
                candidate_args = {}
                if self.layer_id == self.candidate_source_layer:
                    candidate_args = dict(
                        candidate_output=self._candidates[offset:end],
                        candidate_output_lengths=self._candidate_lens[offset:end],
                    )
                elif self.layer_id > self.candidate_source_layer:
                    source = self._context[_source(self.prefix, self.candidate_source_layer)]
                    candidate_args = dict(
                        candidate_indices=source._candidates[offset:end],
                        candidate_lengths=source._candidate_lens[offset:end],
                    )
                index_plan = self._index_plan(index_mode, count)
                scratch = current_workspace_manager().get_simultaneous(
                    *((spec.shape, spec.dtype) for spec in dsa_indexer.scratch_specs(index_plan, device=iq_data.device))
                )
                binding = dsa_indexer.bind(
                    index_plan,
                    scratch=scratch,
                    q_mxfp4=iq_data[offset:end],
                    q_scales=iq_scale[offset:end],
                    query_weights=iw[offset:end],
                    index_k_cache=self.indexer.k_cache.kv_cache,
                    page_table=index_pages[:count],
                    cache_lengths=im.cache_lengths[offset:end],
                    active_width=self._active,
                    score_width=score_width,
                    output_indices=self.topk_indices_buffer[offset:end],
                    **candidate_args,
                )
                retain_cuda_graph_capture_resource(scratch)
                dsa_indexer.score(binding)
                dsa_indexer.select(binding)

        # Reuse the shared arena only after indexing completes. Keep full-batch
        # metadata beside, not overlapping, the native attention scratch.
        metadata_specs = self._attention_workspace_specs[mode]
        arena = current_workspace_manager().get_simultaneous(
            *metadata_specs,
            *((spec.shape, spec.dtype) for spec in state.scratch_plan.scratch_specs()),
        )
        buffers = arena
        metadata_count = len(metadata_specs)
        # Decode layers sharing cache groups map identical pages, so the first
        # layer of each group keeps its metadata for the rest of the forward.
        # Every replay rewrites it before use, so it is an ordinary graph-pool
        # activation: retaining it would pin one copy per captured graph.
        if mode == "decode":
            key = (
                id(swa),
                id(main),
                self.is_draft,
                self.is_ced_decoder,
                rows,
                metadata_specs,
            )
            context = get_forward_context()
            shared = context.__dict__.setdefault("_ds41_decode_metadata", {})
            entry = shared.get(key)
            if entry is None or entry[0] is not swa or entry[1] is not main:
                metadata_buffers = tuple(
                    torch.empty(shape, dtype=dtype, device=q.device) for shape, dtype in metadata_specs
                )
                self._write_decode_metadata(metadata_buffers, swa, main, rows)
                entry = shared[key] = (swa, main, metadata_buffers)
            buffers = [*entry[2], *arena[metadata_count:]]
        else:
            self._write_decode_metadata(buffers, swa, main, rows)
        swa_indices, swa_lengths, top_lengths = buffers[:3]
        kwargs = {}
        if main is not None:
            kwargs = dict(
                indexed_indices=owner.topk_indices_buffer[:rows],
                indexed_lengths=top_lengths[:rows],
                indexed_page_table=buffers[3][:rows],
            )
        binding = mla.bind(
            plan,
            scratch=buffers[metadata_count:],
            q=q,
            swa_indices=swa_indices[:rows],
            swa_lengths=swa_lengths[:rows],
            **kwargs,
        )
        # Keep the borrowed arena, not aliases of the transient query storage.
        retain_cuda_graph_capture_resource(arena)
        mla.run(
            binding=binding,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            indexed_k_cache=self._owner().kv_cache if main is not None else None,
            swa_page_size=self.swa_cache_layer.block_size,
            indexed_page_size=self._main_page,
            sm_scale=512**-0.5,
            attn_sink=self.attn_sink,
            out=output,
            cache_format="deepseek_v41",
        )

    def _write_decode_metadata(self, buffers, swa, main, rows):
        swa_indices, swa_lengths, top_lengths = buffers[:3]
        visible = main.cache_lengths if main is not None else swa.cache_lengths
        _chunk[(rows,)](
            swa.positions,
            swa.req_id_per_token,
            swa.block_table,
            swa_indices,
            swa_lengths,
            top_lengths,
            visible,
            swa.query_start_loc,
            swa.request_positions,
            0,
            swa.block_table.stride(0),
            self.swa_cache_layer.block_size,
            self.window_size,
            self.swa_width,
            self.is_draft,
            triton.next_power_of_2(self.swa_width),
            swa_replay_start=(swa.swa_replay_start if self.is_ced_decoder else swa.swa_prefix_replay_start),
        )
        if main is not None:
            _pages[(rows, triton.cdiv(self._main_width, 128))](
                main.req_id_per_token,
                main.block_table,
                buffers[3],
                0,
                main.block_table.stride(0),
                main.block_table.shape[1],
                self._main_width,
                128,
            )

    def setup_wo_projection(self):
        groups = self.n_local_groups
        heads_per_group = self.n_local_heads // groups
        group_width = heads_per_group * self.head_dim
        rank = self.o_lora_rank
        hidden = self.hidden_size
        expected = (
            ("WO-A weight", self.wo_a.weight, (groups * rank, group_width)),
            (
                "WO-A scale",
                self.wo_a.weight_scale_inv,
                (groups * (rank // 32), group_width // 32),
            ),
            ("WO-B weight", self.wo_b.weight, (hidden, groups * rank)),
            (
                "WO-B scale",
                self.wo_b.weight_scale_inv,
                (hidden // 32, groups * rank // 32),
            ),
        )
        for name, tensor, shape in expected:
            if tuple(tensor.shape) != shape:
                raise RuntimeError(f"V4.1 {name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
        self._wo_projection_weights = wo_projection.pack_weights(
            self.wo_a.weight.detach(),
            self.wo_a.weight_scale_inv.detach(),
            self.wo_b.weight.detach(),
            self.wo_b.weight_scale_inv.detach(),
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
            block_size=(32, 32),
        )
        q_weight, kv_weight = self.q_norm.weight, self.kv_norm.weight
        capacities = getattr(self.wq_b, "b12x_capacities", ())
        consumers = (self.wq_b,) if self.indexer is None else (self.wq_b, self.indexer.wq_b)
        eps = self.q_norm.variance_epsilon
        self._qkv_norm_quant = (
            self.q_lora_rank == 1280
            and q_weight.shape == (1280,)
            and kv_weight.shape == (512,)
            and q_weight.dtype == kv_weight.dtype == torch.bfloat16
            and q_weight.device == kv_weight.device
            and q_weight.device.type == "cuda"
            and q_weight.is_contiguous()
            and kv_weight.is_contiguous()
            and self.q_norm.streams == self.kv_norm.streams == 1
            and eps == self.kv_norm.variance_epsilon
            and math.isfinite(eps)
            and eps > 0
            and bool(capacities)
            and capacities[-1] >= 64
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
            and self.rotary_emb.cos_sin_cache.stride(1) == 1
            and all(
                getattr(layer, "b12x_capacities", ()) == capacities
                and getattr(layer, "b12x_weight", None) is not None
                and layer.b12x_weight.in_features == 1280
                and layer.b12x_weight.block_size == (32, 32)
                and getattr(layer, "bias", None) is None
                for layer in consumers
            )
        )
        if self._qkv_norm_quant:
            prepare_qkv_norm_quant(q_weight.device)

    def _wo_preparation_unit(self, workload):
        from b12x.preparation import PreparedCall

        weights = self._wo_projection_weights
        if weights is None:
            raise PreparationResourceUnavailableError("V4.1 WO weights are not packed")
        table = self.rotary_emb.cos_sin_cache
        device = table.device
        heads_per_group = self.n_local_heads // self.n_local_groups
        nope_dim = self.head_dim - self.rope_head_dim
        rope_dim = self.rope_head_dim

        def prepare(state):
            rows = state.query.max_tokens
            source = torch.empty(
                (rows, self.n_local_heads, self.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            positions = torch.arange(rows, dtype=torch.int64, device=device)
            positions.remainder_(table.shape[0])
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in state._scratch_state.scratch_specs()
            )
            binding = state.bind_inv_rope(
                scratch=scratch,
                o=source,
                positions=positions,
                cos_sin_cache=table,
                weights=weights,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
            )
            return PreparedCall(
                run=lambda: state.run_inv_rope(binding),
                produce=lambda: source.normal_(std=0.25),
                owners=(weights, table),
            )

        requests = []
        decode_rows = self._attention_declarations["decode"].query.query_rows
        token_counts = tuple(
            sorted(
                {
                    *self._preparation_token_counts(workload),
                    *range(1, decode_rows + 1),
                }
            )
        )
        for rows in token_counts:
            requests.append(
                self._wo_plan(rows).request(
                    name=f"{self.prefix}.wo.m{rows}",
                    prepare_call=prepare,
                    benchmark_call=prepare,
                )
            )
        requests.append(
            self._wo_plan(self.capacity, is_prefill=True).request(
                name=f"{self.prefix}.wo.prefill",
                prepare_call=prepare,
                benchmark_call=prepare,
            )
        )
        return B12xPreparationUnit(
            name="V41WOProjection",
            key=(self.prefix, token_counts),
            requests=tuple(requests),
            stage="weights",
        )

    def _wo_plan(self, rows: int, *, is_prefill: bool = False):
        key = "prefill" if is_prefill else rows
        planned_rows = self.capacity if is_prefill else rows
        if key not in self._wo_plans:
            weights = self._wo_projection_weights
            if weights is None:
                raise PreparationResourceUnavailableError("V4.1 WO weights are not packed")
            table = self.rotary_emb.cos_sin_cache
            self._wo_plans[key] = wo_projection.plan(
                wo_projection.Caps(
                    device=table.device,
                    max_tokens=planned_rows,
                    groups=weights.groups,
                    group_width=weights.group_width,
                    rank=weights.rank,
                    hidden=weights.hidden,
                ),
                invocation=dict(
                    operation="inv_rope",
                    dynamic_tokens=is_prefill,
                    heads_per_group=self.n_local_heads // self.n_local_groups,
                    nope_dim=self.head_dim - self.rope_head_dim,
                    rope_dim=self.rope_head_dim,
                    positions_dtype="int64",
                    cos_sin_dtype=str(table.dtype).removeprefix("torch."),
                ),
            )
        return self._wo_plans[key]

    def _o_proj(self, o, positions, *, is_prefill=False):
        from b12x.preparation import require_prepared

        rows = o.shape[0]
        plan = self._wo_plans.get("prefill" if is_prefill else rows)
        if plan is None or plan.prepared is None:
            raise PreparationResourceUnavailableError(
                f"V4.1 WO projection is not prepared for {rows} rows with is_prefill={is_prefill}"
            )
        require_prepared(plan, "gemm.wo_projection", o.device)
        weights = self._wo_projection_weights
        assert weights is not None
        scratch = _scratch(plan)
        binding = wo_projection.bind_inv_rope(
            plan,
            scratch=scratch,
            o=o,
            positions=positions,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            weights=weights,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.head_dim - self.rope_head_dim,
            rope_dim=self.rope_head_dim,
        )
        # The custom attention result must outlive reuse of the workspace arena.
        output = torch.empty_strided(
            (rows, weights.hidden, 1),
            (weights.hidden, 1, rows * weights.hidden),
            dtype=torch.bfloat16,
            device=o.device,
        )
        binding = replace(binding, output=output)
        # The model owns the plan and weights; graph pools manage activation
        # lifetimes. Keep only the borrowed workspace, not layer activations.
        retain_cuda_graph_capture_resource(scratch)
        local = wo_projection.run_inv_rope(
            binding=binding,
            plan=plan,
            stream=current_stream().cuda_stream,
        )
        if local.dtype != torch.bfloat16:
            raise TypeError("V4.1 WO projection must return BF16")
        if get_tensor_model_parallel_world_size() > 1:
            local = get_tp_group().all_reduce(local)
        return local
