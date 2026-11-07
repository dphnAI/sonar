# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA-aware FlashInfer TRT-LLM MoE experts (BF16, MXFP4 x BF16).

Reuses the routed API + ``gemm1_lora_delta`` path from FlashInfer PR #3153
(BF16) and PR #3987 (FP4 block-scale):

* The W13 (gate_up) LoRA delta is passed directly as ``gemm1_lora_delta`` to
  the routed kernel, which fuses it into FC1 before SwiGLU.
* The W2 (down) LoRA cannot be fused. We take the FC1 activation output
  returned by the kernel together with ``expanded_idx_to_permuted_idx``,
  unpermute it, compute the W2 delta out of kernel via punica, and add it to
  the already-finalized output.

Constraints (matching the PR support matrix; final gating lives in the oracle):
* SM100+ (Blackwell), gated SwiGLU, shuffled weights only;
* BF16 weights, or MXFP4 weights with BF16 activations;
* routing must be computed outside the MoE (the Modular path satisfies this).
"""

import functools
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
from aphrodite.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
from aphrodite.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from aphrodite.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    view_as_block_major_k,
)
from aphrodite.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsBase,
)
from aphrodite.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from aphrodite.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
from aphrodite.model_executor.layers.quantization.utils.flashinfer_utils import activation_to_flashinfer_int
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from aphrodite.platforms import current_platform
from aphrodite.triton_utils import tl, triton
from aphrodite.utils.flashinfer import has_flashinfer_trtllm_fused_moe


@triton.jit
def _unpermute_activation_kernel(
    act_ptr,
    idx_ptr,
    out_ptr,
    num_cols,
    stride_ar,
    stride_or,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    col_offs = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    col_mask = col_offs < num_cols

    idx = tl.load(idx_ptr + row)
    out_ptrs = out_ptr + row * stride_or + col_offs
    if idx >= 0:
        vals = tl.load(act_ptr + idx * stride_ar + col_offs, mask=col_mask, other=0.0)
        tl.store(out_ptrs, vals, mask=col_mask)
    else:
        zeros = tl.zeros((BLOCK_I,), dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptrs, zeros, mask=col_mask)


@triton.jit
def _finalize_lora_kernel(
    gemm2_ptr,
    weight_ptr,
    idx_ptr,
    delta_ptr,
    out_ptr,
    K,
    stride_g0,
    stride_d0,
    stride_d1,
    stride_o0,
    scale,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = col < K

    acc_base = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_delta = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        eid = token * TOP_K + k
        pidx = tl.load(idx_ptr + eid)
        if pidx >= 0:
            w = tl.load(weight_ptr + eid).to(tl.float32)
            base = tl.load(gemm2_ptr + pidx * stride_g0 + col, mask=mask, other=0.0).to(tl.float32)
            acc_base += w * base
        acc_delta += tl.load(delta_ptr + token * stride_d0 + k * stride_d1 + col, mask=mask, other=0.0).to(tl.float32)

    out = acc_base * scale + acc_delta
    tl.store(out_ptr + token * stride_o0 + col, out.to(out_ptr.dtype.element_ty), mask=mask)


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
        # apply() writes the fully finalized result into `output`.
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
        topk_ids_and_weights: tuple[torch.Tensor, torch.Tensor],
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
        activation: MoEActivation,
    ) -> list[torch.Tensor]:
        """Call the dtype-specific routed MoE and return its tensors.

        The LoRA path sets gemm1_lora_delta and runs with do_finalize=False so
        the base finalize can be fused with the W2 LoRA reduction.

        Return contract:
        gemm1_lora_delta is None -> [output] (do_finalize=True)
        otherwise                -> [gemm2_output(permuted, unweighted),
                                     expert_weights,
                                     expanded_idx_to_permuted_idx,
                                     gemm1_activation_output(permuted)]
        """
        raise NotImplementedError

    def prepare_gemm1_lora_delta(
        self,
        gemm1_lora_delta: torch.Tensor,
        lora_context: MoELoRAContext,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Hook to rewrite the W13 delta before it is handed to the kernel."""
        return gemm1_lora_delta

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
        # LoRA buffers use the padded hidden size; `output` may be narrower.
        K = self.hidden_dim

        # ---- Base-model fast path ----
        # When no token in the batch selects a LoRA adapter, skip the LoRA machinery
        # and run the plain base MoE with do_finalize=True, which writes the finalized
        # result straight into `output`.
        if self._batch_has_no_lora(lora_context):
            self.invoke_routed_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_ids_and_weights=(topk_ids, topk_weights),
                gemm1_lora_delta=None,  # without LoRA, no delta
                global_num_experts=global_num_experts,
                a1q_scale=a1q_scale,
                output=output,
                activation=activation,
            )
            return

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

        # Zero-filled because under EP, non-local (token, top_k) slots must
        # stay 0 so they contribute no bias when fed to FlashInfer.
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
            swap_w13_slices=True,
        )
        gemm1_lora_delta = self.prepare_gemm1_lora_delta(gemm1_lora_delta, lora_context, topk_ids)

        ret = self.invoke_routed_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_ids_and_weights=(topk_ids, topk_weights),
            gemm1_lora_delta=gemm1_lora_delta,
            global_num_experts=global_num_experts,
            a1q_scale=a1q_scale,
            output=output,
            activation=activation,
        )

        gemm2_permuted = ret[0]
        expert_weights = ret[1]
        expanded_idx_to_permuted_idx = ret[2]
        gemm1_act_permuted = ret[3]
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
        self._finalize_with_w2_lora(
            output,
            gemm2_permuted,
            expert_weights,
            expanded_idx_to_permuted_idx,
            w2_delta,
            num_tokens,
            top_k,
            scale=1.0,
        )

    @staticmethod
    def _batch_has_no_lora(lora_context: MoELoRAContext) -> bool:
        meta = getattr(lora_context.punica_wrapper, "token_mapping_meta", None)
        if meta is None:
            return False
        flag = meta.no_lora_flag_cpu
        return bool(flag.numel() == 1 and flag.item())

    @staticmethod
    def _unpermute_activation(
        act_permuted: torch.Tensor,
        idx_map: torch.Tensor,
        num_tokens: int,
        top_k: int,
        intermediate_size: int,
    ) -> torch.Tensor:
        """Permuted FC1 activation -> (num_tokens * top_k, intermediate_size)."""
        num_rows = num_tokens * top_k
        out = torch.empty(
            (num_rows, intermediate_size),
            dtype=act_permuted.dtype,
            device=act_permuted.device,
        )
        block_i = 1024
        grid = (num_rows, triton.cdiv(intermediate_size, block_i))
        _unpermute_activation_kernel[grid](
            act_permuted,
            idx_map,
            out,
            intermediate_size,
            act_permuted.stride(0),
            out.stride(0),
            BLOCK_I=block_i,
        )
        return out

    @staticmethod
    def _finalize_with_w2_lora(
        output: torch.Tensor,
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        idx_map: torch.Tensor,
        w2_delta: torch.Tensor,
        num_tokens: int,
        top_k: int,
        scale: float = 1.0,
    ) -> None:
        """Fused base finalize + W2 LoRA reduction, written into ``output``."""
        # Inputs may carry hidden padding that `output` does not.
        K = output.size(1)
        assert gemm2_permuted.size(1) >= K and w2_delta.size(2) >= K
        block_k = 512
        grid = (num_tokens, triton.cdiv(K, block_k))
        _finalize_lora_kernel[grid](
            gemm2_permuted,
            expert_weights.reshape(-1),
            idx_map,
            w2_delta,
            output,
            K,
            gemm2_permuted.stride(0),
            w2_delta.stride(0),
            w2_delta.stride(1),
            output.stride(0),
            scale,
            TOP_K=top_k,
            BLOCK_K=block_k,
        )


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
        topk_ids_and_weights: tuple[torch.Tensor, torch.Tensor],
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
        activation: MoEActivation,
    ) -> list[torch.Tensor]:
        import flashinfer
        from flashinfer.fused_moe import WeightLayout

        do_finalize = gemm1_lora_delta is None
        ret = flashinfer.fused_moe.trtllm_bf16_routed_moe(
            topk_ids=topk_ids_and_weights,
            hidden_states=hidden_states,
            gemm1_weights=view_as_block_major_k(w1),
            gemm2_weights=view_as_block_major_k(w2),
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
            do_finalize=do_finalize,
            output=output if do_finalize else None,
            activation_type=activation_to_flashinfer_int(activation),
        )
        if not do_finalize:
            return list(ret)
        return [output]


# MXFP4 weights x BF16 activations trtllm MoE + LoRA
class TrtLlmMxfp4LoRAExperts(_TrtLlmLoRAExpertsBase, TrtLlmMxfp4ExpertsBase):
    """MXFP4-weight, BF16-activation trtllm MoE + LoRA. Not MXFP8 activations:
    the kernel then returns the FC1 activation quantized, without its scales."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        _TrtLlmLoRAExpertsBase.__init__(self, moe_config, quant_config)
        # gemm1_alpha / gemm1_beta / gemm1_clamp_limit, hidden_dim_unpadded.
        TrtLlmMxfp4ExpertsBase.__init__(self, moe_config, quant_config)

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kMxfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SWIGLUOAI, MoEActivation.SILU)

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        return True  # routing is done outside the kernel

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # DP gathers tokens, but the no-LoRA fast path reads the local batch.
        p = moe_parallel_config
        return p.dp_size == 1 and not p.use_all2all_kernels and not p.enable_eplb

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        # flashinfer manages its own workspace and drops the hidden padding.
        return (0,), (0,), (M, self.hidden_dim_unpadded)

    def output_hidden_dim(self, hidden_dim: int) -> int:
        # FusedMoEExpertsModular precedes TrtLlmMxfp4ExpertsBase in the MRO.
        return self.hidden_dim_unpadded

    @functools.cached_property
    def _w13_bias_up_gate_index(self) -> torch.Tensor:
        """Columns of the kernel-format FC1 bias in [up | gate] order."""
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

        # Kernel format is the (up_i, gate_i) bias permuted for the tile.
        assert self.w1_bias is not None
        bias = self.w1_bias.reshape(self.local_num_experts, -1)[0]
        permute = get_w2_permute_indices_with_cache({}, bias.reshape(-1, 1), 128)
        interleaved = torch.argsort(permute.to(bias.device))
        return torch.cat([interleaved[0::2], interleaved[1::2]])

    def prepare_gemm1_lora_delta(
        self,
        gemm1_lora_delta: torch.Tensor,
        lora_context: MoELoRAContext,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Bring the W13 delta to [up | gate] halves and fold in the FC1 bias."""
        intermediate_size = gemm1_lora_delta.size(-1) // 2
        # View as (I, 2) pairs of the first and second w13 half. SWIGLUOAI
        # weights, and the deltas that follow them, are interleaved.
        if self.moe_config.activation == MoEActivation.SWIGLUOAI:
            pairs = gemm1_lora_delta.unflatten(-1, (intermediate_size, 2))
        else:
            pairs = gemm1_lora_delta.unflatten(-1, (2, intermediate_size)).mT
        # Per-expert deltas arrive as (up, gate); fused 3D ones follow the
        # checkpoint (gate, up) order.
        if lora_context.w13_num_slices != 2:
            pairs = pairs.flip(-1)
        delta = pairs.mT.flatten(-2)  # [up | gate]

        # The kernel has a single FC1 bias slot, shared by the static bias and
        # the per-token delta, so the static bias rides along in the delta.
        if self.w1_bias is not None:
            bias = self.w1_bias.reshape(self.local_num_experts, -1)
            bias = bias[:, self._w13_bias_up_gate_index].float()
            local_ids = topk_ids.long() - self.ep_rank * self.local_num_experts
            local_ids = local_ids.clamp_(0, self.local_num_experts - 1)
            delta = (delta.float() + bias[local_ids]).to(torch.bfloat16)
        return delta.contiguous()

    def invoke_routed_moe(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids_and_weights: tuple[torch.Tensor, torch.Tensor],
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
        activation: MoEActivation,
    ) -> list[torch.Tensor]:
        from flashinfer import trtllm_fp4_block_scale_routed_moe

        assert a1q_scale is None and hidden_states.dtype == torch.bfloat16
        topk_ids, topk_weights = topk_ids_and_weights
        # See TrtLlmBf16LoRAExperts.invoke_routed_moe for the return contract.
        do_finalize = gemm1_lora_delta is None
        ret = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids.to(torch.int32), topk_weights),
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_weights=w1,
            gemm1_weights_scale=self.w1_scale,
            # With a delta the bias is folded into it (prepare_gemm1_lora_delta).
            gemm1_bias=self.w1_bias if do_finalize else None,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.w2_scale,
            gemm2_bias=self.w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            # Routing is already done; Renormalize is a safe default.
            routing_method_type=RoutingMethodType.Renormalize,
            do_finalize=do_finalize,
            enable_pdl=True,
            activation_type=self._flashinfer_activation_type(activation),
            output=output if do_finalize else None,
            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
            gemm1_lora_delta=gemm1_lora_delta,
        )
        return [output] if do_finalize else list(ret)
