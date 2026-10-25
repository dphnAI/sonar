# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn as nn

from aphrodite.config import AphroditeConfig
from aphrodite.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from aphrodite.forward_context import get_forward_context, is_forward_context_available
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from aphrodite.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from aphrodite.model_executor.model_loader.weight_utils import default_weight_loader
from aphrodite.model_executor.models.interfaces import (
    EagleModelMixin,
)
from aphrodite.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
)
from aphrodite.models.deepseek_v41.common.mm_preprocess import image_sentinel_mask
from aphrodite.models.deepseek_v41.nvidia.model import (
    DeepseekV4MoE,
    MegaGateRoutingMetadata,
    _make_deepseek_v4_weights_mapper,
    make_deepseek_v4_expert_params_mapping,
    prepare_mega_gate_routing_metadata,
)
from aphrodite.models.deepseek_v41.nvidia.model import (
    DeepseekV41LLMForCausalLM as UpstreamDeepseekV41LLMForCausalLM,
)
from aphrodite.sequence import IntermediateTensors
from aphrodite.utils.b12x import (
    b12x_layer_prefix,
    register_b12x_layer,
    set_b12x_preparation_provider,
)
from aphrodite.v1.attention.backends.registry import AttentionBackendEnum

from .attention import DeepseekV41B12xAttention
from .b12x_layers import B12xMHC
from .b12x_layers import B12xRMSNorm as RMSNorm
from .ced import ced_decoder_start, gather_rows, scatter_rows
from .engram import Engram, EngramLayout, NgramHashState
from .mhc_split import join_mhc_split
from .sparse_mla import DeepseekV41B12xMetadata

logger = init_logger(__name__)


def _select_dsv4_attn_cls(aphrodite_config):
    backend = aphrodite_config.attention_config.backend
    if backend not in (None, AttentionBackendEnum.B12X_DSV41):
        raise ValueError("Native DeepSeek V4.1 accepts only B12X_DSV41 attention")
    return DeepseekV41B12xAttention


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        aphrodite_config,
        prefix,
        topk_indices_buffer: torch.Tensor,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = Engram(
                    config,
                    aphrodite_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    prefix=f"{prefix}.engram",
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(aphrodite_config)(
            aphrodite_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
        )
        if config.scoring_func != "sqrtsoftplus" or not config.norm_topk_prob:
            raise ValueError("V4.1 requires normalized sqrtsoftplus routing")
        if getattr(config, "gate_temp", 1.0) != 1.0:
            raise ValueError("V4.1 requires gate_temp=1")
        self.ffn = DeepseekV4MoE(
            aphrodite_config,
            prefix=f"{prefix}.ffn",
            use_sequence_parallel=False,
        )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self._b12x_mhc = B12xMHC(config)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
        mega_gate_metadata: MegaGateRoutingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_output = x if residual is not None else None
        apply_engram = self.engram is not None and engram_hashes is not None
        if residual is None:
            residual = x
        elif apply_engram:
            # Engram mutates the reconstructed residual; do not fuse across it.
            residual = self._b12x_mhc.post(x, residual, post_mix, res_mix)
            previous_output = None
        if self.engram is not None and engram_hashes is not None:
            if residual.ndim == 2:
                residual = residual[:, None, :].expand(-1, self.hc_mult, -1).contiguous()
            residual = self.engram(residual, engram_hashes[:, self.engram.layer_hash_index], engram_mask)
        fn = self.hc_attn_fn_broadcast if residual.ndim == 2 else self.hc_attn_fn
        residual, post_mix, res_mix, x, attn_pre = self._b12x_mhc.pre(
            residual,
            fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm.weight,
            pre_mix,
            previous_output=previous_output,
            previous_post=post_mix if previous_output is not None else None,
            previous_comb=res_mix if previous_output is not None else None,
        )
        global_kv_ready = None
        if ced_indices is not None:
            # The coefficient gathers below read the split's side-stream outputs.
            join_mhc_split(x.device)
            # Decoder global KV needs every encoder row, but decoder queries,
            # residual updates and experts only need the selected replay rows.
            global_kv_ready = self.attn.prepare_global_kv(positions, x)
            x = gather_rows(x, ced_indices)
            residual = gather_rows(residual, ced_indices)
            post_mix = gather_rows(post_mix, ced_indices)
            res_mix = gather_rows(res_mix, ced_indices)
            attn_pre = gather_rows(attn_pre, ced_indices)
            positions = gather_rows(positions, ced_indices)
            if input_ids is not None:
                input_ids = gather_rows(input_ids, ced_indices)
        x = self.attn(positions, x, global_kv_ready=global_kv_ready)
        residual, post_mix, res_mix, x, ffn_pre = self._b12x_mhc.post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm.weight,
            attn_pre,
        )
        x = self.ffn(x, input_ids, mega_gate_metadata=mega_gate_metadata)
        return x, residual, post_mix, res_mix, ffn_pre


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()

        config = aphrodite_config.model_config.hf_config
        quant_config = aphrodite_config.quant_config
        self.config = config
        self.aphrodite_config = aphrodite_config
        self.ced_decoder_start = ced_decoder_start(config)
        self.quant_config = quant_config
        parallel = aphrodite_config.parallel_config
        if parallel.decode_context_parallel_size != 1:
            raise NotImplementedError("Native DeepSeek V4.1 does not support DCP")
        if parallel.prefill_context_parallel_size != 1:
            raise NotImplementedError("Native DeepSeek V4.1 does not support PCP")
        if aphrodite_config.parallel_config.use_ubatching:
            raise ValueError("V4.1 fixed native layer buffers require ubatching/DBO disabled")
        if aphrodite_config.lora_config is not None:
            raise ValueError("V4.1 native kernels do not support LoRA adapters")
        if quant_config is None or quant_config.get_name() != "deepseek_v41_fp8":
            raise ValueError("V4.1 requires its native block32 quantization config")
        self.vocab_size = config.vocab_size
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            aphrodite_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                aphrodite_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )

        # Hashing reads the runner's accepted-only lookback, never KV slot state.
        self.engram_hash: NgramHashState | None = None
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            local_engram = any(
                isinstance(layer, DeepseekV4DecoderLayer) and layer.engram is not None
                for layer in islice(self.layers, self.start_layer, self.end_layer)
            )
            if local_engram:
                first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(self.engram_layout)
                self.engram_swa_prefix = swa_cache_module.prefix
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        spec_config = aphrodite_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                aphrodite_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=aphrodite_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the final hc collapse — plus the
        # (num_tokens, hc_mult) pre-mix the next rank's first layer needs
        # for its attention collapse.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if self.engram_hash is not None and input_ids is not None:
            attn_metadata = get_forward_context().attn_metadata if is_forward_context_available() else None
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[0]
            if isinstance(attn_metadata, dict):
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(DeepseekV41B12xMetadata, attn_metadata[self.engram_swa_prefix])
                query_start_loc = swa_metadata.query_start_loc
                if lookback_token_ids is None:
                    raise ValueError("V4.1 Engram requires accepted-only lookback_token_ids")
            else:
                # Profiling must exercise the same native hash/lookup/projection
                # and gate kernels, not silently omit the Engram layers.
                query_start_loc = torch.zeros(2, dtype=torch.int32, device=input_ids.device)
                query_start_loc[1] = input_ids.shape[0]
                lookback_token_ids = input_ids.new_full((1, 3), -1)
            image_mask = image_sentinel_mask(input_ids)
            engram_mask = ~image_mask
            engram_hashes = self.engram_hash(
                input_ids,
                query_start_loc,
                image_mask,
                lookback_token_ids,
            )
            for layer in islice(self.layers, self.start_layer, self.end_layer):
                engram = getattr(layer, "engram", None)
                if engram is not None:
                    engram.prepare_embeddings(engram_hashes[:, engram.layer_hash_index])

        full_num_tokens = positions.shape[0]
        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        decoder_compacted = False
        if ced_indices is not None and self.ced_decoder_start is not None and self.start_layer > self.ced_decoder_start:
            # Pipeline transport keeps its original full-row ABI. A decoder
            # stage gathers only the rows materialized by the preceding stage.
            hidden_states = gather_rows(hidden_states, ced_indices)
            pre_mix = gather_rows(pre_mix, ced_indices)
            positions = gather_rows(positions, ced_indices)
            if input_ids is not None:
                input_ids = gather_rows(input_ids, ced_indices)
            decoder_compacted = True
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None  # avoid duplicate mhc_post call
        mega_gate_metadata = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            boundary_indices = ced_indices if idx == self.ced_decoder_start else None
            if layer.ffn.use_native_mega_moe and (mega_gate_metadata is None or boundary_indices is not None):
                routing_ids = gather_rows(input_ids, boundary_indices) if boundary_indices is not None else input_ids
                if routing_ids is None:
                    raise ValueError("Native MegaMoE routing requires input_ids")
                mega_gate_metadata = prepare_mega_gate_routing_metadata(
                    routing_ids,
                    has_hash_routing=False,
                    image_sentinel_base_id=(
                        layer.ffn.image_sentinel_lo if layer.ffn.gate.bias_vl is not None else None
                    ),
                )
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
                ced_indices=boundary_indices,
                mega_gate_metadata=mega_gate_metadata,
            )
            if boundary_indices is not None:
                positions = gather_rows(positions, ced_indices)
                if input_ids is not None:
                    input_ids = gather_rows(input_ids, ced_indices)
                decoder_compacted = True
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models
                aux_recon = layer._b12x_mhc.post(hidden_states, residual, post_mix, res_mix)
                aux_hidden_state = layer._b12x_mhc.collapse(aux_recon)
                if decoder_compacted:
                    aux_hidden_state = scatter_rows(aux_hidden_state, ced_indices, full_num_tokens)
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
        if layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
            if self.end_layer in self.aux_hidden_state_layers:
                hidden_states = final_aux_recon
            else:
                hidden_states = layer._b12x_mhc.post(hidden_states, residual, post_mix, res_mix)

        if not get_pp_group().is_last_rank:
            join_mhc_split(hidden_states.device)
            if decoder_compacted:
                hidden_states = scatter_rows(hidden_states, ced_indices, full_num_tokens)
                pre_mix = scatter_rows(pre_mix, ced_indices, full_num_tokens)
            return IntermediateTensors({"hidden_states": hidden_states, "pre_mix": pre_mix})

        # Keep full HC states for MTP before collapsing the output.
        if self._mtp_hidden_buffer is not None:
            buffer_states = (
                scatter_rows(hidden_states, ced_indices, full_num_tokens) if decoder_compacted else hidden_states
            )
            self._mtp_hidden_buffer[:full_num_tokens].copy_(buffer_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        hidden_states = layer._b12x_mhc.collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if decoder_compacted:
            hidden_states = scatter_rows(hidden_states, ced_indices, full_num_tokens)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if ".engram.embed_tokens." in name:
                if is_pp_missing_parameter(name, self):
                    continue
                module_name, _, leaf = name.rpartition(".")
                embedding = self.get_submodule(module_name)
                loaded_params.update(
                    f"{module_name}.{loaded}" for loaded in embedding.load_weights(((leaf, loaded_weight),))
                )
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    default_weight_loader(params_dict[name][:n], narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # ``<head>.base_layer.<leaf>``; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = self.layers[self.start_layer]
        if first_layer.ffn.use_native_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            _finalize_mhc_broadcast_weights(layer)


def _finalize_mhc_broadcast_weights(layer: DeepseekV4DecoderLayer) -> None:
    broadcast = layer.hc_attn_fn.detach().view(-1, layer.hc_mult, layer.hidden_size).sum(dim=1)
    if layer.hc_attn_fn_broadcast is None:
        layer.hc_attn_fn_broadcast = broadcast
    else:
        layer.hc_attn_fn_broadcast.copy_(broadcast)


class DeepseekV41LLMForCausalLM(UpstreamDeepseekV41LLMForCausalLM):
    model_cls = DeepseekV4Model

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__(aphrodite_config=aphrodite_config, prefix=prefix)
        self.hf_to_aphrodite_mapper = _make_deepseek_v4_weights_mapper(
            getattr(self.config, "expert_dtype", "fp4"), "weight_scale_inv"
        )

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    supports_b12x_native_preparation = True

    @property
    def decoder_replay_layers(self):
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        # AutoWeightsLoader may revisit this child for another contiguous
        # prefix group. Finalize only in the root model's post-load hook.
        return loader.load_weights(weights, mapper=self.hf_to_aphrodite_mapper)

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mhc_broadcast_weights()
        for module in self.modules():
            if isinstance(module, DeepseekV4DecoderLayer):
                set_b12x_preparation_provider(module, module._b12x_mhc)
                name = b12x_layer_prefix(module)
                register_b12x_layer(name, module)
                module._b12x_mhc.bind_layer_name(name)
                module._b12x_mhc.prepare_split(module.hc_attn_fn.device)
            if isinstance(module, DeepseekV41B12xAttention):
                module.setup_wo_projection()
            if isinstance(module, Engram):
                module.process_weights_after_loading()
