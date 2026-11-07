# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM110 grouped MXFP6 experts implemented with CuTe DSL."""

import torch

import aphrodite.model_executor.layers.fused_moe.modular_kernel as mk
from aphrodite.model_executor.kernels.linear.mxfp6.cutedsl_grouped import (
    cutedsl_grouped_mxfp6_gemm,
)
from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
)
from aphrodite.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    get_moe_permute_scratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from aphrodite.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from aphrodite.model_executor.layers.fused_moe.utils import _resize_cache
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from aphrodite.platforms import current_platform


class CutedslMxfp6Experts(mk.FusedMoEExpertsModular):
    """Standard-layout MXFP6 MoE experts for Blackwell SM110."""

    def __init__(self, *args, weight_format: str, activation_format: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_format = weight_format
        self.activation_quant_format = activation_format

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and current_platform.is_device_capability_family(110)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key in {
            kMxfp6E2M3Static,
            kMxfp6E3M2Static,
        } and activation_key in {
            kMxfp8Dynamic,
            kMxfp6E2M3Dynamic,
            kMxfp6E3M2Dynamic,
        }

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(config: FusedMoEParallelConfig) -> bool:
        # The standard path handles TP. EP requires an expert-map-aware grouped
        # metadata builder and is deliberately rejected for the first backend.
        return config.ep_size == 1

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim >= 128 and hidden_dim % 128 == 0

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        del global_num_experts, local_num_experts, expert_tokens_meta
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        return (
            (M * topk, max(N, K)),
            (M * topk, max(activation_out_dim, K)),
            (M, K),
        )

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        if not moe_permute_unpermute_supported():
            return None
        return get_moe_permute_scratch(
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
            device=torch.device(self.moe_config.device),
        )

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
        del a1q_scale, a2_scale, expert_tokens_meta
        if apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise ValueError("router-weight-on-input requires top-k=1")
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        m, k = hidden_states.shape
        topk = topk_ids.shape[1]
        n = self.moe_config.intermediate_size_per_partition
        local_experts = w1.shape[0]
        num_experts = local_experts if global_num_experts < 0 else global_num_experts
        if expert_map is not None:
            num_experts = expert_map.numel()
        permuted = _resize_cache(workspace2, (m * topk, k))
        permuted, _, offsets, inv_perm, _ = moe_permute(
            hidden_states,
            None,
            topk_ids,
            num_experts,
            local_experts,
            expert_map,
            permuted_hidden_states=permuted,
            scratch=self._get_permute_scratch(),
        )

        gate_up = _resize_cache(workspace13, (m * topk, 2 * n))
        cutedsl_grouped_mxfp6_gemm(
            permuted,
            w1,
            self.w1_scale,
            offsets,
            gate_up,
            2 * n,
            k,
            self.activation_quant_format,
            self.weight_format,
        )
        activated = _resize_cache(workspace2, (m * topk, n))
        self.activation(activation, activated, gate_up)

        down = _resize_cache(workspace13, (m * topk, k))
        cutedsl_grouped_mxfp6_gemm(
            activated,
            w2,
            self.w2_scale,
            offsets,
            down,
            k,
            n,
            self.activation_quant_format,
            self.weight_format,
        )
        moe_unpermute(
            out=output,
            permuted_hidden_states=down,
            topk_weights=torch.ones_like(topk_weights) if apply_router_weight_on_input else topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=offsets,
        )
