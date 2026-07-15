# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA-aware FlashInfer TRT-LLM MoE experts (BF16).

Reuses the routed API + ``gemm1_lora_delta`` path from FlashInfer PR #3153:

* The W13 (gate_up) LoRA delta is passed directly as ``gemm1_lora_delta`` to
  the routed kernel, which fuses it into FC1 before SwiGLU.
* The W2 (down) LoRA cannot be fused. We take the FC1 activation output
  returned by the kernel together with ``expanded_idx_to_permuted_idx``,
  unpermute it, compute the W2 delta out of kernel via punica, and add it to
  the already-finalized output.

Constraints (matching the PR support matrix; final gating lives in the oracle):

* SM100+ (Blackwell), gated SwiGLU, shuffled weights only;
* BF16 only;
* routing must be computed outside the MoE (the Modular path satisfies this).
"""

from abc import abstractmethod

import torch

import aphrodite.model_executor.layers.fused_moe.modular_kernel as mk
from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from aphrodite.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from aphrodite.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from aphrodite.model_executor.layers.fused_moe.utils import (
    trtllm_moe_pack_topk_ids_weights,
)
from aphrodite.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from aphrodite.model_executor.layers.quantization.utils.quant_utils import QuantKey
from aphrodite.platforms import current_platform
from aphrodite.utils.flashinfer import has_flashinfer_trtllm_fused_moe


class _TrtLlmLoRAExpertsBase(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """LoRA-aware TRT-LLM MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = moe_config.intermediate_size_per_partition
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer_trtllm_fused_moe()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # FlashInfer manages its own workspace; only declare output (M, K).
        return (0,), (0,), (M, K)

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """Override the base 3D-weight assumption.

        FusedMoEKernel._fused_experts calls moe_problem_size before apply(),
        but the base impl asserts ``len(w1.shape) == 3``. The FlashInfer
        TRT-LLM path stores shuffled weights in 4D BlockMajorK layout, so we
        derive the (E, M, N, K, topk) tuple from config + inputs instead.
        The N/K here only feed workspace sizing, which we zero out in
        workspace_shapes(); the real shapes are handled inside FlashInfer.
        """
        E = self.local_num_experts
        N = 2 * self.intermediate_size_per_partition
        K = self.hidden_dim
        M = a1.size(0) if a1.dim() == 2 else a1.size(1)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @abstractmethod
    def invoke_routed_moe(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        packed_topk_ids: torch.Tensor,
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
        activation: MoEActivation,
    ) -> list[torch.Tensor]:
        """Call the dtype-specific routed MoE and return its tensors.

        Return contract (do_finalize=True):
        gemm1_lora_delta is None -> [output]
        otherwise                -> [output, expanded_idx_to_permuted_idx,
                                     gemm1_activation_output(permuted)]
        """
        raise NotImplementedError

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        lora_context = self._lora_context
        assert lora_context is not None, "LoRA context must be set"
        num_tokens = hidden_states.size(0)
        top_k = self.topk
        intermediate_size = self.intermediate_size_per_partition
        K = output.size(1)

        # The LoRA tile-config heuristic unpacks w1/w2 as standard 3D MoE
        # weights, but FlashInfer stores shuffled 4D BlockMajorK weights.
        # add_lora_w13/add_lora_w2 only read .shape from w1/w2, so pass
        # zero-storage meta tensors carrying the logical 3D shapes.
        w1_cfg = torch.empty(
            (self.local_num_experts, 2 * intermediate_size, K),
            device="meta",
            dtype=torch.bfloat16,
        )
        w2_cfg = torch.empty(
            (self.local_num_experts, K, intermediate_size),
            device="meta",
            dtype=torch.bfloat16,
        )

        packed_topk_ids = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)

        gemm1_lora_delta = torch.zeros(
            num_tokens,
            top_k,
            2 * intermediate_size,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

        lora_x = hidden_states
        if not self.expects_unquantized_inputs:
            orig = lora_context.original_hidden_states
            assert orig is not None and orig.shape[0] == hidden_states.shape[0], (
                "quantized TRT-LLM LoRA path requires original_hidden_states"
            )
            lora_x = orig

        # add_inputs=False: write the pure delta only. The base is fused in by
        # the kernel and routing weights are not applied before SwiGLU.
        w13_meta = self.apply_w13_lora(
            lora_context,
            y=gemm1_lora_delta,
            x=lora_x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=expert_map,
            w1=w1_cfg,
            w2=w2_cfg,
            num_tokens=num_tokens,
            top_k_num=top_k,
            add_inputs=False,
        )

        # vLLM/Aphrodite W13 order is [gate, up], while FlashInfer expects
        # [up, gate] for gemm1_lora_delta.
        gemm1_lora_delta = torch.cat(
            [
                gemm1_lora_delta[..., intermediate_size:],
                gemm1_lora_delta[..., :intermediate_size],
            ],
            dim=-1,
        )

        ret = self.invoke_routed_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            packed_topk_ids=packed_topk_ids,
            gemm1_lora_delta=gemm1_lora_delta,
            global_num_experts=global_num_experts,
            a1q_scale=a1q_scale,
            output=output,
            activation=activation,
        )

        expanded_idx_to_permuted_idx = ret[1]
        gemm1_act_permuted = ret[2]
        act = self._unpermute_activation(
            gemm1_act_permuted,
            expanded_idx_to_permuted_idx,
            num_tokens,
            top_k,
            intermediate_size,
        )

        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = w13_meta

        w2_delta = torch.zeros(
            num_tokens,
            top_k,
            K,
            dtype=output.dtype,
            device=output.device,
        )
        self.apply_w2_lora(
            lora_context,
            y=w2_delta,
            x=act,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=w1_cfg,
            w2=w2_cfg,
            top_k_num=top_k,
            add_inputs=False,
        )
        output.add_(w2_delta.sum(dim=1))

    @staticmethod
    def _unpermute_activation(
        act_permuted: torch.Tensor,
        idx_map: torch.Tensor,
        num_tokens: int,
        top_k: int,
        intermediate_size: int,
    ) -> torch.Tensor:
        """Permuted FC1 activation -> (num_tokens * top_k, intermediate_size)."""
        valid = idx_map >= 0
        safe_idx = idx_map.clamp_min(0).long()
        gathered = act_permuted[safe_idx]
        return gathered * valid.unsqueeze(1).to(act_permuted.dtype)


class TrtLlmBf16LoRAExperts(_TrtLlmLoRAExpertsBase):
    """BF16 unquantized TRT-LLM MoE + LoRA."""

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU]

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    def invoke_routed_moe(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        packed_topk_ids: torch.Tensor,
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
        activation: MoEActivation,
    ) -> list[torch.Tensor]:
        import flashinfer
        from flashinfer.fused_moe import WeightLayout

        ret = flashinfer.fused_moe.trtllm_bf16_routed_moe(
            topk_ids=packed_topk_ids,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm2_weights=w2,
            gemm1_lora_delta=gemm1_lora_delta,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=self.routing_method_type,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(activation),
        )
        if isinstance(ret, (list, tuple)):
            output.copy_(ret[0])
            return list(ret)
        output.copy_(ret)
        return [output]
