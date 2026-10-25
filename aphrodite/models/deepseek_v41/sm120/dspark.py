# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4.1 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the
DSV4.1 DSpark, which reuses the target model's architecture similarly to MTP.
Its weights ship in the target checkpoint under the ``mtp.{0,1,2}.*`` prefix.

To implement non-causal attention, we leverage the sparse attention implementation to
include the future query tokens in the top-k indices for each query token.
"""

from bisect import bisect_left
from collections.abc import Iterable
from types import SimpleNamespace

import regex as re
import torch
import torch.nn as nn
from b12x.gemm import block_fp8_linear

from aphrodite.config import AphroditeConfig, get_current_aphrodite_config
from aphrodite.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from aphrodite.model_executor.layers.linear import ReplicatedLinear
from aphrodite.model_executor.layers.logits_processor import LogitsProcessor
from aphrodite.model_executor.model_loader.weight_utils import default_weight_loader
from aphrodite.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from aphrodite.model_executor.models.utils import maybe_prefix
from aphrodite.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    b12x_layer_prefix,
    register_b12x_layer,
    register_b12x_unit_provider,
    set_b12x_preparation_provider,
)
from aphrodite.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

from .b12x_layers import B12xRMSNorm as RMSNorm
from .b12x_layers import (
    _execution_capacities,
)
from .model import (
    DeepseekV4DecoderLayer,
    _finalize_mhc_broadcast_weights,
    make_deepseek_v4_expert_params_mapping,
    prepare_mega_gate_routing_metadata,
)

logger = init_logger(__name__)

# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class _ContextKVProjection:
    """Project and normalize all draft context KVs with one input quantization."""

    def __init__(self, attns: Iterable[nn.Module], capacity: int) -> None:
        attns = tuple(attns)
        fused = attns[0].fused_wqa_wkv
        starts = tuple(attn.q_lora_rank for attn in attns)
        if any(start % 32 for start in starts):
            raise ValueError("DSpark context KV must start on a checkpoint scale block")
        weights = tuple(attn.fused_wqa_wkv.weight[start:] for attn, start in zip(attns, starts))
        scales = tuple(attn.fused_wqa_wkv.weight_scale_inv[start // 32 :] for attn, start in zip(attns, starts))
        self.weight = block_fp8_linear.pack_weight(
            torch.cat([weight.view(torch.uint8) for weight in weights]).view(weights[0].dtype),
            torch.cat([scale.view(torch.uint8) for scale in scales]).view(scales[0].dtype),
            block_size=(32, 32),
        )
        self.num_layers = len(attns)
        self.norm = RMSNorm(
            attns[0].kv_norm.weight.numel(),
            attns[0].kv_norm.variance_epsilon,
            streams=self.num_layers,
        )
        self.norm.weight = nn.Parameter(torch.cat([attn.kv_norm.weight for attn in attns]), requires_grad=False)
        self.capacities = tuple(
            sorted(
                {
                    capacity,
                    *(bound for bound in _execution_capacities() if bound <= capacity),
                }
            )
        )
        self.plans = tuple(
            block_fp8_linear.plan(
                block_fp8_linear.Caps(
                    device=fused.weight.device,
                    max_tokens=bound,
                    in_features=fused.weight.shape[1],
                    out_features=self.weight.out_features,
                    block_size=(32, 32),
                    output_mode="provided",
                )
            )
            for bound in self.capacities
        )
        register_b12x_unit_provider(self)

    def get_b12x_preparation_units(self, layer: object, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        if self.weight.weight.values.is_meta:
            return ()

        def make_call(state, *, bound: int):
            from b12x.preparation import PreparedCall

            source = torch.zeros(
                (bound, self.weight.in_features),
                dtype=torch.bfloat16,
                device=state.device,
            )
            output = torch.empty(
                (bound, self.weight.out_features, 1),
                dtype=torch.bfloat16,
                device=source.device,
            )
            scratch = [
                torch.empty(spec.shape, dtype=spec.dtype, device=source.device)
                for spec in state.scratch.scratch_specs()
            ]
            binding = state.bind(
                scratch=scratch,
                source=source,
                packed_weight=self.weight,
                output=output,
            )
            return PreparedCall(
                run=lambda: state.run_binding(binding),
                produce=lambda: source.normal_(std=0.25),
                owners=(self.weight,),
            )

        requests = tuple(
            plan.request(
                name=f"dspark.context_kv.{id(self):x}.m{bound}",
                prepare_call=lambda state, bound=bound: make_call(state, bound=bound),
                benchmark_call=lambda state, bound=bound: make_call(state, bound=bound),
            )
            for bound, plan in zip(self.capacities, self.plans)
        )
        return self.norm.get_b12x_preparation_units(self.norm, workload) + (
            B12xPreparationUnit(
                name="DSparkContextKV",
                key=(id(self), self.capacities),
                requests=requests,
                stage="weights",
            ),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        index = bisect_left(self.capacities, x.shape[0])
        if index == len(self.capacities):
            raise ValueError("DSpark context rows exceed prepared capacity")
        rows = x.shape[0]
        plan = self.plans[index]
        out = torch.empty(
            (rows, self.weight.out_features),
            dtype=torch.bfloat16,
            device=x.device,
        )
        scratch = current_workspace_manager().get_simultaneous(
            *((spec.shape, spec.dtype) for spec in plan.scratch_specs())
        )
        binding = block_fp8_linear.bind(
            plan,
            scratch=scratch,
            source=x,
            packed_weight=self.weight,
            output=out.view(rows, self.weight.out_features, 1),
        )
        retain_cuda_graph_capture_resource(scratch)
        block_fp8_linear.run(binding=binding)
        return self.norm(out).view(rows, self.num_layers, self.weight.out_features // self.num_layers)


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "") -> None:
        super().__init__()
        assert aphrodite_config.speculative_config is not None
        config = aphrodite_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.context_capacity = aphrodite_config.scheduler_config.max_num_batched_tokens
        self._context_kv_projection: _ContextKVProjection | None = None

        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None) or getattr(config, "num_nextn_predict_layers", None) or 3
        )

        # Bound from the loaded target, as in K3DSparkModel. Do not create an
        # unused vocabulary allocation while loading the draft checkpoint.
        self.embed_tokens: nn.Module | None = None

        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=aphrodite_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.topk_indices_buffer = torch.empty(
            aphrodite_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        current_aphrodite_config = get_current_aphrodite_config()
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_aphrodite_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        # Heads: final norm, and the Markov + confidence heads.
        # Loaded from the "final" MTP layer weights (mtp.*) in the target
        # checkpoint. v4.1 has no learned hc_head: the hc copies are
        # collapsed with a pre-mix derived from the last layer's hc_ffn
        # projection at the end of forward().
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states)).

        ``aux_hidden_states`` is [T, hidden_size * len(target_layer_ids)].
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Insert the sliding-window context KV for every draft layer.

        Mirrors the reference DSparkAttention: each layer derives its context KV
        from the SAME projected target hidden ``main_x``, via that layer's own
        ``wkv`` + ``kv_norm`` + RoPE + quant, then writes it at the
        layer's context slots.

        ``context_slot_mappings`` is either common to all layers or a per-layer
        list (each entry is the mapping for that layer's kv-cache group).
        ``None`` (or a ``None`` entry) runs the projection to reserve workspace
        but writes nothing (profiling).
        """
        assert self._context_kv_projection is not None
        kvs = self._context_kv_projection(main_x)
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mappings
                if context_slot_mappings is None or isinstance(context_slot_mappings, torch.Tensor)
                else context_slot_mappings[i]
            )
            attn = layer.attn
            kv = kvs[:, i]
            if slot_mapping is None:
                continue
            attn.insert_context_kv(kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # The first layer's post-load broadcast weights consume [T, H] directly.
        hidden_states = inputs_embeds
        mega_gate_metadata = None
        first_ffn = self.layers[0].ffn
        if first_ffn.use_native_mega_moe:
            mega_gate_metadata = prepare_mega_gate_routing_metadata(
                input_ids,
                has_hash_routing=False,
                image_sentinel_base_id=(first_ffn.image_sentinel_lo if first_ffn.gate.bias_vl is not None else None),
            )

        residual = post_mix = res_mix = pre_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                mega_gate_metadata=mega_gate_metadata,
            )
        hidden_states = layer._b12x_mhc.post(hidden_states, residual, post_mix, res_mix)
        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference forward_head applies (hc_pre with the
        # last block's ffn pre-mix). Return the PRE-norm head hidden;
        # compute_logits applies self.norm.
        assert pre_mix is not None
        hidden_states = layer._b12x_mhc.collapse(hidden_states, pre_mix)
        return hidden_states


class DSparkDeepseekV4ForCausalLM(nn.Module):
    # Draft weights ship in the target checkpoint (mtp.*) without embed/head, so
    # load_dspark_model always aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Full-vocab draft: draft ids are target ids, no remapping needed.
    draft_id_to_target_id = None
    supports_b12x_native_preparation = True

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "") -> None:
        super().__init__()
        assert aphrodite_config.speculative_config is not None
        self.draft_model_config = aphrodite_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = aphrodite_config.quant_config
        self.linear_scale_name = "weight_scale_inv"
        self.model = DSparkDeepseekV4Model(aphrodite_config=aphrodite_config, prefix=maybe_prefix(prefix, "model"))
        # Bound by load_dspark_model after loading. An unused online-quantized
        # head otherwise remains on meta and cannot be natively precompiled.
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- Hooks used by the speculator -------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # DSV4 MLA path: each draft layer's sliding-window cache is a separate
        # layer, named by its prefix.
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mappings)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Returns the pre-norm collapsed head hidden ([T, hidden_size]).
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    @staticmethod
    def sample_draft_with_bias(
        logits, bias, idx_map, temperature, seeds, positions, cache, column, *, use_fp64=False
    ) -> torch.Tensor | None:
        from .sampling import sample_rounded_sum

        if (
            logits.dtype != bias.dtype
            or logits.dtype != cache.dtype
            or logits.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or logits.stride(-1) != 1
            or bias.stride(-1) != 1
            or cache.stride(-1) != 1
        ):
            return None
        return sample_rounded_sum(
            logits, bias, idx_map, temperature, seeds, positions, cache, column, use_fp64=use_fp64
        )

    def draft_vocab_shard(self):
        """This rank's vocab shard of the aliased target lm_head, or None."""
        from aphrodite.v1.worker.gpu.sample.vocab_parallel import vocab_shard_for

        if self.lm_head is None:
            return None
        return vocab_shard_for(self.lm_head)

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's base-logit shard: the gathered path's per-rank GEMM, no gather."""
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states), skip_gather=True)

    def markov_bias_local(self, markov_embed: torch.Tensor, shard) -> torch.Tensor:
        """Markov bias for global tokens [shard.start, shard.end) only.

        Restricts the replicated markov_w2 to the shard's rows and projects
        through the same `_apply_head` arithmetic as the full bias.
        """
        head = self.model.markov_head.markov_w2
        rows = SimpleNamespace(weight=head.weight[shard.start : shard.end], quant_method=head.quant_method)
        return self.logits_processor(rows, markov_embed, skip_gather=True)

    @staticmethod
    def sample_draft_with_bias_vocab_parallel(
        logits, bias, idx_map, temperature, seeds, positions, cache, column, shard, *, use_fp64=False
    ) -> torch.Tensor:
        """Fused rounded-sum draft sampling on this rank's shard; global ids."""
        from .sampling import sample_rounded_sum_vocab_parallel

        return sample_rounded_sum_vocab_parallel(
            logits, bias, idx_map, temperature, seeds, positions, cache, column, shard, use_fp64=use_fp64
        )

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

        Non-mtp weights (embed/head/main layers) belong to the target model and
        are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
        """
        # Draft MoE layers use the dspark_* expert counts, not the
        # backbone's (see DeepseekV4MoE and the reference
        # ModelArgs.get_moe_config).
        n_draft_experts = getattr(self.config, "dspark_n_routed_experts", 0) or self.config.n_routed_experts
        if self.model.layers[0].ffn.use_native_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(n_draft_experts)
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=n_draft_experts,
            )
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
        )

        # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_confidence_head = False

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped
            if "confidence_head." in name:
                loaded_confidence_head = True

            # ``.scale`` -> per-method scale suffix.
            if name.endswith(".scale"):
                suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else f".{self.linear_scale_name}"
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")

            # E8M0 expert scales: keep raw exponent bytes.
            if ".experts." in name:
                if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack params
            # (main_proj/norm/markov_head/confidence_head) load directly —
            # otherwise e.g. "markov_w1" would collide with the "w1" shard rule.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    default_weight_loader(params_dict[name][: narrow.shape[0]], narrow)
                    loaded_params.add(name)
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if self.model.confidence_head is not None and not loaded_confidence_head:
            self.model.confidence_head = None
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers:
            layer.attn.setup_wo_projection()
        self.model._context_kv_projection = _ContextKVProjection(
            (layer.attn for layer in self.model.layers), self.model.context_capacity
        )
        _finalize_mhc_broadcast_weights(self.model.layers[0])
        for layer in self.model.layers:
            mhc = layer._b12x_mhc
            set_b12x_preparation_provider(layer, mhc)
            name = b12x_layer_prefix(layer)
            register_b12x_layer(name, layer)
            mhc.bind_layer_name(name)

    def _remap_dspark_name(self, name: str) -> str | None:
        """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

        Returns None for non-mtp weights (owned by the target model).
        """
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        if rest.startswith("confidence_head.") and self.model.confidence_head is None:
            return None
        # The checkpoint calls the Markov head's factors ``embed``/``head``;
        # DSparkMarkovHead registers them as ``markov_w1``/``markov_w2``.
        if rest.startswith("markov_head.embed."):
            return "model.markov_head.markov_w1." + rest.removeprefix("markov_head.embed.")
        if rest.startswith("markov_head.head."):
            return "model.markov_head.markov_w2." + rest.removeprefix("markov_head.head.")
        # Head-stack params live at model level (mtp.last), context combiner at
        # model level (mtp.0); everything else is a per-layer decoder block.
        head_prefixes = (
            "norm.",
            "markov_head.",
            "confidence_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(head_prefixes):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"
