# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swordfish fused MoE experts (Blackwell sm100/sm110 wNa16)."""

import functools

import torch

import aphrodite._custom_ops as ops
import aphrodite.model_executor.layers.fused_moe.modular_kernel as mk
from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from aphrodite.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from aphrodite.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt8Static,
)
from aphrodite.model_executor.layers.quantization.utils.swordfish_utils import (
    SWORDFISH_BLOCK_N,
)
from aphrodite.platforms import current_platform

# Token-block size fed to moe_align_block_size and the kernel. 16-token
# blocks run the per-warp Stream-K kernel that wins small batches; 32-token
# blocks fuse two m16 tiles per staged weight chunk, amortizing the dequant
# once experts average enough tokens to fill them.
SWORDFISH_MOE_BLOCK_SIZE = 16
SWORDFISH_MOE_BLOCK_SIZE_BATCHED = 32
# Average tokens per expert above which the batched block size pays off.
SWORDFISH_MOE_BATCHED_THRESHOLD = 32
# Average tokens per expert above which sorting the tokens once and running
# the tcgen05 prefill GEMM per expert segment beats the fused kernel. Pays
# one host sync for the segment bounds, acceptable in prefill. The serial
# per-expert launches underfill large parts, so the path only engages on
# small ones (Thor class); big parts stay on the fused 32-token blocks.
SWORDFISH_MOE_GROUPED_THRESHOLD = 128
SWORDFISH_MOE_GROUPED_MAX_SMS = 40


@functools.cache
def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


class SwordfishExperts(mk.FusedMoEExpertsModular):
    """Swordfish-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        assert quant_config.use_int4_w4a16 or quant_config.use_int8_w8a16, "Supports only int4_w4a16 or int8_w8a16"
        assert quant_config.w1_zp is None and quant_config.w2_zp is None, (
            "Swordfish MoE v1 supports only symmetric quantization (no zero points)"
        )
        assert quant_config.w1_bias is None and quant_config.w2_bias is None, "Swordfish MoE v1 does not support bias"
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.gemm1_alpha = quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        self.gemm1_beta = quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0

        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        if not (p.is_cuda() and p.has_device_capability(100)):
            return False
        # sm100 family only, datacenter 10.x and Thor 11.x. Consumer
        # Blackwell (12.x) is a different SM and is untested.
        capability = p.get_device_capability()
        return capability is not None and capability.major in (10, 11)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key in [kInt4Static, kInt8Static]

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Gated activations only; moe_problem_size derives the intermediate
        # size from the packed w1 assuming the 2N gate/up layout. Activation
        # itself goes through the shared apply_moe_activation() callback.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.SWIGLUSTEP,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % 64 == 0

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def num_bits(self) -> int:
        return 4 if self.quant_config.use_int4_w4a16 else 8

    @property
    def group_size(self) -> int:
        block_shape = self.quant_config.block_shape
        return block_shape[1] if block_shape is not None else -1

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Swordfish packed weights are [E, N/64, K/64, words], with w1 holding
        # the gate/up shards along N.
        assert w1.dim() == 4 and w2.dim() == 4

        E = w1.size(0)
        K = a1.size(-1)
        N = (w1.size(1) * SWORDFISH_BLOCK_N) // 2

        assert a1.dim() == 2
        # Make sure we are using the correct a1 (pre-permute).
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

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
        # swordfish_moe_mm allocates its own GEMM outputs, so the workspaces
        # only back the final output buffer provisioned by the modular kernel.
        workspace1 = (M * topk, K)
        workspace2 = (M * topk, K)
        output = (M, K)
        return (workspace1, workspace2, output)

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
    ):
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dtype in [torch.float16, torch.bfloat16]
        assert topk_weights.dtype == torch.float32
        assert activation.is_gated

        E, M, N, K, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)

        if global_num_experts == -1:
            global_num_experts = E
        if (
            M * topk >= SWORDFISH_MOE_GROUPED_THRESHOLD * E
            and _sm_count() <= SWORDFISH_MOE_GROUPED_MAX_SMS
            and expert_map is None
            and not apply_router_weight_on_input
            and hidden_states.dtype == torch.bfloat16
            and self.num_bits == 4
            and K % 128 == 0
            and 2 * N % 128 == 0
            and N % 128 == 0
        ):
            self._apply_grouped(
                output, hidden_states, w1, w2, topk_weights, topk_ids,
                activation, M, N, K, topk, apply_router_weight_on_input,
            )
            return
        block_size = (
            SWORDFISH_MOE_BLOCK_SIZE_BATCHED
            if M * topk >= SWORDFISH_MOE_BATCHED_THRESHOLD * E
            else SWORDFISH_MOE_BLOCK_SIZE
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        num_bits = self.num_bits
        group_size = self.group_size

        intermediate_cache1 = ops.swordfish_moe_mm(
            hidden_states,
            w1,
            self.w1_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights if apply_router_weight_on_input else None,
            block_size,
            topk,
            apply_router_weight_on_input,
            num_bits,
            group_size,
            K,
            2 * N,
        )

        intermediate_cache2 = torch.empty(
            (M * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.activation(
            activation,
            intermediate_cache2,
            intermediate_cache1.view(-1, 2 * N),
            clamp_limit=self.gemm1_clamp_limit,
            alpha=self.gemm1_alpha,
            beta=self.gemm1_beta,
        )

        intermediate_cache3 = ops.swordfish_moe_mm(
            intermediate_cache2,
            w2,
            self.w2_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            None if apply_router_weight_on_input else topk_weights,
            block_size,
            1,
            not apply_router_weight_on_input,
            num_bits,
            group_size,
            N,
            K,
        )

        ops.moe_sum(intermediate_cache3.view(M, topk, K), output)

    def _apply_grouped(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        M: int,
        N: int,
        K: int,
        topk: int,
        apply_router_weight_on_input: bool,
    ) -> None:
        E = w1.size(0)
        group_size = self.group_size
        flat = topk_ids.flatten().to(torch.int64)
        order = torch.argsort(flat, stable=True)
        bounds = torch.cumsum(torch.bincount(flat, minlength=E), 0).cpu().tolist()
        a_sorted = hidden_states.index_select(0, order // topk).contiguous()

        cache1 = torch.empty((M * topk, 2 * N), dtype=hidden_states.dtype, device=hidden_states.device)
        beg = 0
        for e, end in enumerate(bounds):
            if end > beg:
                cache1[beg:end] = ops.swordfish_prefill_mm(
                    a_sorted[beg:end], w1[e], self.w1_scale[e], group_size, K, 2 * N
                )
            beg = end
        cache2 = torch.empty((M * topk, N), dtype=hidden_states.dtype, device=hidden_states.device)
        self.activation(
            activation,
            cache2,
            cache1,
            clamp_limit=self.gemm1_clamp_limit,
            alpha=self.gemm1_alpha,
            beta=self.gemm1_beta,
        )
        out_sorted = torch.empty((M * topk, K), dtype=hidden_states.dtype, device=hidden_states.device)
        beg = 0
        for e, end in enumerate(bounds):
            if end > beg:
                out_sorted[beg:end] = ops.swordfish_prefill_mm(
                    cache2[beg:end], w2[e], self.w2_scale[e], group_size, N, K
                )
            beg = end
        wt = topk_weights.flatten().index_select(0, order).unsqueeze(1)
        out_sorted = out_sorted * wt.to(out_sorted.dtype)
        out_flat = torch.zeros((M * topk, K), dtype=out_sorted.dtype, device=out_sorted.device)
        out_flat.index_copy_(0, order, out_sorted)
        ops.moe_sum(out_flat.view(M, topk, K), output)
