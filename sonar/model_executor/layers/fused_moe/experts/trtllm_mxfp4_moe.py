# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from typing import TYPE_CHECKING, Any

import torch

import sonar.envs as envs
import sonar.model_executor.layers.fused_moe.modular_kernel as mk
from sonar.model_executor.layers.fused_moe.activation import MoEActivation
from sonar.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from sonar.model_executor.layers.fused_moe.moe_output import (
    UnfinalizedMoEOutput,
    convert_flashinfer_moe_output,
)
from sonar.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from sonar.model_executor.layers.fused_moe.utils import (
    fi_moe_largest_bucket,
)
from sonar.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
    has_flashinfer_situ_activation,
)
from sonar.model_executor.layers.quantization.utils.mxfp4_scale_compression import (
    MAX_SELECTED_EXPERTS,
)
from sonar.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from sonar.platforms import current_platform
from sonar.utils.flashinfer import has_flashinfer
from sonar.utils.torch_utils import current_stream
from sonar.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from flashinfer.fused_moe.core import TRTLLMCanonicalRouting

    from sonar.model_executor.layers.quantization.utils.mxfp4_scale_compression import (
        CompressedMXFP4Scales,
    )


def _run_prepared_mxfp4(canonical: "TRTLLMCanonicalRouting", **kwargs: Any) -> list[torch.Tensor]:
    """Reuse canonical routing when the tuned body uses the same routing tile."""
    from flashinfer import trtllm_fp4_block_scale_routed_moe
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.core import (
        TrtllmMoERoutingMetadataSlot,
        get_trtllm_moe_sm100_module,
    )
    from flashinfer.fused_moe.da_moe import DABody
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.tllm_enums import (
        Fp8QuantizationType,
        RoutingInputMode,
        SfLayout,
        WeightLayout,
        deduce_trtllm_gen_tensor_dtype,
    )

    tuner = AutoTuner.get()
    manager = current_workspace_manager()
    # Graphs retain their prepared body. Eager calls after locking can use
    # token counts that were never captured.
    if tuner.is_tuning_mode or manager.is_locked():
        return trtllm_fp4_block_scale_routed_moe(**kwargs)
    x, x_scale = kwargs["hidden_states"], kwargs["hidden_states_scale"]
    num_tokens, hidden_size = x.shape
    top_k = kwargs["top_k"]
    output = kwargs["output"]
    stream = current_stream().cuda_stream
    if output is None:
        output = manager.get_persistent(("mxfp4_prepared_output", stream, num_tokens), (num_tokens, 0), torch.bfloat16)
    ids, weights = kwargs["topk_ids"]
    inputs = MoeRunnerInputs(
        output=output,
        routing_logits=None,
        topk_ids=ids,
        expert_weights=weights,
        hidden_states=x,
        hidden_states_scale=x_scale,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    module = get_trtllm_moe_sm100_module()
    runner = module.MoERunner(
        module.moe_op,
        top_k=top_k,
        num_local_experts=kwargs["local_num_experts"],
        dtype_act=deduce_trtllm_gen_tensor_dtype(x, x_scale),
        dtype_weights=deduce_trtllm_gen_tensor_dtype(kwargs["gemm1_weights"], kwargs["gemm1_weights_scale"]),
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=hidden_size,
        hidden_size_output=hidden_size,
        intermediate_size=kwargs["intermediate_size"],
        activation_type=kwargs["activation_type"],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=kwargs["num_experts"],
        num_fused_shared_experts=0,
    )
    tuning_config = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=kwargs["tune_max_num_tokens"],
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        hidden_states_scale_layout=SfLayout.layout_linear,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    native_kwargs = dict(
        kwargs,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        num_fused_shared_experts=0,
        enable_pdl=True,
        norm_topk_prob=True,
    )
    flat_inputs = inputs.to_list()
    _, tactic = tuner.choose_one(
        "flashinfer::trtllm_fp4_block_scale_moe", [runner], tuning_config, flat_inputs, **native_kwargs
    )
    if tactic == -1 or tactic[0] != canonical.tile_n:
        return trtllm_fp4_block_scale_routed_moe(**kwargs)

    # CanonicalRoutingBuffers exposes the same native routing maps as the body ABI.
    counts, total, expanded, permuted, histogram, cta_batch, cta_limit, cta_count = canonical.scratch
    metadata = TrtllmMoERoutingMetadataSlot(
        tile_n=canonical.tile_n,
        total_num_padded_tokens=total,
        expanded_idx_to_permuted_idx=expanded[: num_tokens * top_k],
        permuted_idx_to_token_idx=permuted,
        expert_weights=weights,
        expert_count_histogram=histogram,
        num_tokens_per_expert=counts,
        cta_idx_xy_to_batch_idx=cta_batch,
        cta_idx_xy_to_mn_limit=cta_limit,
        num_non_exiting_ctas=cta_count,
    )
    body = DABody(tactic=tactic[1], tile_n=tactic[0])
    adapter = module.DABodyRunner(runner)

    def allocate_workspace():
        return tuple(
            tensor if isinstance(tensor, torch.Tensor) else torch.from_dlpack(tensor)
            for tensor in adapter.prepare_body(flat_inputs, body, metadata, **native_kwargs)
        )

    body_workspace = manager.get_persistent_resource(
        (
            "mxfp4_prepared_body",
            stream,
            num_tokens,
            hidden_size,
            kwargs["num_experts"],
            kwargs["local_num_experts"],
            kwargs["intermediate_size"],
            top_k,
            runner.dtype_act,
            runner.dtype_weights,
            kwargs["activation_type"],
            kwargs["do_finalize"],
            tuple(tactic),
        ),
        allocate_workspace,
    )
    adapter.forward_from_metadata(flat_inputs, body, metadata, body_workspace, **native_kwargs)
    if kwargs["do_finalize"]:
        return [output]
    # FP4DABodyBuffers stores the unfinalized GEMM2 output in its fifth field.
    return [body_workspace[4], weights, expanded[: num_tokens * top_k]]


class TrtLlmMxfp4ExpertsBase:
    """MXFP4 TRTLLM-Gen MoE kernels. Shared base for modular and monolithic."""

    kept_constants: tuple[str, ...] = ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        **kwargs,
    ):
        self.moe_config = moe_config
        self.quant_config = quant_config
        self._compressed_scales: CompressedMXFP4Scales | None = None

        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = moe_config.intermediate_size_per_partition
        self.hidden_dim = moe_config.hidden_dim
        self.hidden_dim_unpadded = moe_config.hidden_dim_unpadded or moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

        # MXFP4-specific TRTLLM parameters from quant_config
        device = torch.accelerator.current_device_index()
        if quant_config.gemm1_alpha is not None:
            self.gemm1_alpha = torch.tensor(
                [quant_config.gemm1_alpha] * self.local_num_experts,
                dtype=torch.float32,
                device=device,
            )
        else:
            self.gemm1_alpha = None

        if quant_config.gemm1_beta is not None:
            self.gemm1_beta = torch.tensor(
                [quant_config.gemm1_beta] * self.local_num_experts,
                dtype=torch.float32,
                device=device,
            )
        else:
            self.gemm1_beta = None

        if quant_config.gemm1_clamp_limit is not None:
            self.gemm1_clamp_limit = torch.tensor(
                [quant_config.gemm1_clamp_limit] * self.local_num_experts,
                dtype=torch.float32,
                device=device,
            )
        else:
            self.gemm1_clamp_limit = None

        # SITU (SituGLU) TRTLLM-Gen kernel computes
        #   left  = alpha * tanh(x0 / alpha) * sigmoid(x0)   # gate (x0)
        #   right = beta  * tanh(x1 / beta)                  # up   (x1)
        # which matches Sonar's situ_and_mul with (beta, linear_beta), so map
        # situ beta -> gatedActAlpha (gemm1_alpha) and situ linear_beta ->
        # gatedActBeta (gemm1_beta). Both must be > 0.
        if moe_config.activation == MoEActivation.SITU:
            situ_beta = moe_config.activation_situ_beta
            situ_linear_beta = moe_config.activation_situ_linear_beta
            assert situ_beta is not None and situ_beta > 0, "SITU requires activation_situ_beta > 0"
            assert situ_linear_beta is not None and situ_linear_beta > 0, (
                "TRTLLM SiTuGlu requires activation_situ_linear_beta > 0 (the private cubin has no up-passthrough path)"
            )
            self.gemm1_alpha = torch.full(
                (self.local_num_experts,),
                float(situ_beta),
                dtype=torch.float32,
                device=device,
            )
            self.gemm1_beta = torch.full(
                (self.local_num_experts,),
                float(situ_linear_beta),
                dtype=torch.float32,
                device=device,
            )
            self.gemm1_clamp_limit = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_mxfp4_scales_compressed", False):
            raise NotImplementedError("Compressed MXFP4 scales do not support weight reloads. Restart the engine.")
        if not envs.SONAR_MXFP4_COMPRESS_SCALES:
            return
        if self.moe_config.moe_parallel_config.enable_eplb:
            raise ValueError("MXFP4 scale compression does not support expert load balancing.")
        from sonar.model_executor.layers.quantization.utils.mxfp4_scale_compression import (
            encode_mxfp4_scales,
        )

        w1_scale, w2_scale = self.quant_config.w1_scale, self.quant_config.w2_scale
        assert w1_scale is not None and w2_scale is not None
        self._scale_dtypes = (w1_scale.dtype, w2_scale.dtype)
        codec = encode_mxfp4_scales(w1_scale, w2_scale)
        if codec.encoded_nbytes >= codec.raw_nbytes:
            return
        for name in ("packed", "offsets", "literals"):
            layer.register_buffer(f"mxfp4_scale_{name}", getattr(codec, name))
        self._compressed_scales = codec
        layer._mxfp4_scales_compressed = True
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None
        layer.w13_weight_scale = None
        layer.w2_weight_scale = None

    def _weight_scales(
        self,
        expert_ids: torch.Tensor | None = None,
        expert_ids_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codec = self._compressed_scales
        if codec is None:
            w1_scale, w2_scale = self.quant_config.w1_scale, self.quant_config.w2_scale
            assert w1_scale is not None and w2_scale is not None
            return w1_scale, w2_scale
        from flashinfer.autotuner import AutoTuner

        scratch = current_workspace_manager().get_persistent(
            ("mxfp4_scales", current_stream().cuda_stream, codec.w13_shape, codec.w2_shape),
            (codec.raw_nbytes,),
            torch.uint8,
        )
        if AutoTuner.get().is_tuning_mode:
            # Profiling synthesizes routes that can select any local expert.
            if expert_ids_out is not None:
                assert expert_ids is not None
                expert_ids_out.copy_(expert_ids)
            expert_ids = expert_ids_out = None
        elif expert_ids is not None and expert_ids.numel() > MAX_SELECTED_EXPERTS:
            expert_ids = None
        codec.decode_into(
            scratch,
            expert_ids,
            expert_offset=self.ep_rank * self.local_num_experts,
            expert_ids_out=expert_ids_out,
        )
        w1_scale, w2_scale = codec.destination_views(scratch)
        return w1_scale.view(self._scale_dtypes[0]), w2_scale.view(self._scale_dtypes[1])

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer()

    def output_hidden_dim(self, hidden_dim: int) -> int:
        return self.hidden_dim_unpadded

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        if activation == MoEActivation.SITU:
            return has_flashinfer_situ_activation()
        return activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SILU,
        )

    @staticmethod
    def _flashinfer_activation_type(activation: MoEActivation) -> int:
        return activation_to_flashinfer_int(activation)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return False


class TrtLlmMxfp4ExpertsMonolithic(TrtLlmMxfp4ExpertsBase, mk.FusedMoEExpertsMonolithic):
    """Monolithic version of the MXFP4 TRTLLM kernel (router + experts).
    Wraps flashinfer.trtllm_fp4_block_scale_moe().
    """

    def supports_routing_replay_capture(self) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        # Kernel converts to bfloat16 internally
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
        routing_replay_out: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        from flashinfer import trtllm_fp4_block_scale_moe
        from flashinfer.tllm_enums import SfLayout

        if a1q_scale is not None:
            x_quant = hidden_states
            x_scale = a1q_scale.view(torch.float8_e4m3fn)
        else:
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None
        num_tokens = hidden_states.shape[0]
        defer = self.moe_config.should_defer_moe_finalize(num_tokens)
        finalized_output = None
        if not defer:
            finalized_output = torch.empty(
                *hidden_states.shape[:-1],
                self.hidden_dim_unpadded,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

        moe = trtllm_fp4_block_scale_moe
        routing: dict[str, Any] = {
            "routing_logits": router_logits,
            "routing_bias": e_score_correction_bias,
            "routing_replay_out": routing_replay_out,
        }
        if self._compressed_scales is not None and num_tokens * self.topk <= MAX_SELECTED_EXPERTS:
            from flashinfer.fused_moe.core import (
                allocate_trtllm_moe_canonical_routing,
                canonicalize_trtllm_moe_routing_,
            )

            def allocate_routing():
                capacity = MAX_SELECTED_EXPERTS // self.topk
                canonical = allocate_trtllm_moe_canonical_routing(
                    router_logits.new_empty((capacity, global_num_experts)),
                    top_k=self.topk,
                    tile_n=8,
                )
                ids = torch.empty((capacity, self.topk), dtype=torch.int32, device=router_logits.device)
                return canonical, ids

            canonical, ids = current_workspace_manager().get_persistent_resource(
                ("mxfp4_routing", current_stream().cuda_stream, global_num_experts, self.topk),
                allocate_routing,
            )
            canonicalize_trtllm_moe_routing_(
                canonical,
                router_logits,
                e_score_correction_bias,
                x_quant,
                top_k=self.topk,
                n_group=num_expert_group,
                topk_group=topk_group,
                local_expert_offset=self.ep_rank * self.local_num_experts,
                local_num_experts=self.local_num_experts,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=self.routing_method_type,
                use_routing_scales_on_input=apply_router_weight_on_input,
                use_deep_seek_fp8=False,
                norm_topk_prob=True,
                enable_pdl=True,
            )
            selected = canonical.routing_replay_ids[:num_tokens]
            ids = ids[:num_tokens]
            w1_scale, w2_scale = self._weight_scales(selected, ids)
            if routing_replay_out is not None:
                routing_replay_out.copy_(selected)
            routing = {
                "topk_ids": (ids, canonical.expert_weights[:num_tokens]),
                "routing_bias": None,
            }
            moe = partial(_run_prepared_mxfp4, canonical)
            routed_scaling_factor = None
        else:
            w1_scale, w2_scale = self._weight_scales()
        flashinfer_output = moe(
            **routing,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            hidden_states_scale_layout=SfLayout.layout_linear,
            gemm1_weights=w1,
            gemm1_weights_scale=w1_scale,
            gemm1_bias=self.w1_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            gemm2_bias=self.w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=(num_expert_group or 0),
            topk_group=(topk_group or 0),
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            do_finalize=not defer,
            activation_type=self._flashinfer_activation_type(activation),
            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
            output=finalized_output,
        )
        routed_output = convert_flashinfer_moe_output(
            flashinfer_output,
            do_finalize=not defer,
            num_tokens=num_tokens,
            top_k=self.topk,
            finalized_output=finalized_output,
        )
        return routed_output


class TrtLlmMxfp4ExpertsModularBase(TrtLlmMxfp4ExpertsBase, mk.FusedMoEExpertsModular):
    """Modular base version of the MXFP4 TRTLLM kernel (just the experts).
    Wraps flashinfer.trtllm_fp4_block_scale_routed_moe().

    The TRTLLM-Gen kernel skips padding rows (topk_ids == -1), so every
    subclass consumes the padded contiguous layout DeepEP v2 produces.
    """

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.PaddedStandard

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        **kwargs,
    ):
        super().__init__(moe_config, quant_config, **kwargs)
        # Each launch permutes into its own buffer, so a chunked call finalizes.
        moe_config.limit_deferred_moe_finalize(self._max_supported_tokens(self.topk, moe_config.num_experts))

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Modular kernel handles only the expert computation;
        # routing is done externally, so accept any routing method.
        return True

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
        # The workspaces for this implementation are managed by flashinfer.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, self.hidden_dim_unpadded)
        return (workspace1, workspace2, output)

    def _max_supported_tokens(self, top_k: int, global_num_experts: int) -> int:
        """Max tokens per kernel call before the batched-GEMM grid overflows.

        The TRTLLM-Gen batched GEMM launches a static grid whose batch (Y)
        dimension is `getMaxNumCtasInBatchDim(num_tokens, top_k, num_experts,
        tileTokensDim)` and must stay <= 65535. Solving that for num_tokens
        with the smallest tile the kernel may pick (tileTokensDim=8, the runner
        default) gives a bound that is safe regardless of the tactic selected.
        Without it, large batches (e.g. Kimi-K3 top_k=16, EP16 profiling with
        131072 gathered tokens) overflow the grid and the GEMM launch fails.
        """
        MAX_GRID_Y = 65535
        MIN_TILE_TOKENS_DIM = 8
        max_tokens = (MAX_GRID_Y - global_num_experts) * MIN_TILE_TOKENS_DIM // top_k
        return max(1, min(300000, max_tokens))

    def _invoke_kernel(
        self,
        output: torch.Tensor | None,
        x_quant: torch.Tensor,
        x_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        local_num_experts: int,
        local_expert_offset: int,
        topk: int,
    ) -> UnfinalizedMoEOutput | None:
        """Finalize into `output`, or stop after GEMM2 when it is None."""
        from flashinfer import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import SfLayout

        w1_scale, w2_scale = self._weight_scales(topk_ids)
        flashinfer_output = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids, topk_weights),
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            hidden_states_scale_layout=SfLayout.layout_linear,
            gemm1_weights=w1,
            gemm1_weights_scale=w1_scale,
            gemm1_bias=self.w1_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            gemm2_bias=self.w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=global_num_experts,
            top_k=topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=None,
            # Modular kernel receives pre-routed tokens, so routing is already
            # done. Use Renormalize as a safe default the TRTLLM kernel supports.
            routing_method_type=RoutingMethodType.Renormalize,
            do_finalize=output is not None,
            enable_pdl=True,
            activation_type=self._flashinfer_activation_type(activation),
            output=output,
            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
        )
        if output is not None:
            return None
        routed_output = convert_flashinfer_moe_output(
            flashinfer_output,
            do_finalize=False,
            num_tokens=x_quant.shape[0],
            top_k=topk,
        )
        assert isinstance(routed_output, UnfinalizedMoEOutput)
        return routed_output

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
    ) -> UnfinalizedMoEOutput | None:
        topk_ids = topk_ids.to(dtype=torch.int32)

        topk = topk_ids.size(-1)
        local_num_experts = w1.size(0)
        local_expert_offset = self.moe_config.ep_rank * local_num_experts

        if a1q_scale is not None:
            x_quant = hidden_states
            x_scale = a1q_scale.view(torch.float8_e4m3fn)
        else:
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None

        # Chunk tokens so the batched-GEMM grid stays within CUDA limits. Each
        # launch permutes into its own buffer, so only a run that fits in one
        # launch can leave the top-k reduction to a deferring consumer.
        M = x_quant.size(0)
        chunk_size = self._max_supported_tokens(topk, global_num_experts)
        defer = chunk_size >= M and self.moe_config.should_defer_moe_finalize(M)
        unfinalized: UnfinalizedMoEOutput | None = None
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            unfinalized = self._invoke_kernel(
                None if defer else output[start:end],
                x_quant[start:end],
                None if x_scale is None else x_scale[start:end],
                topk_ids[start:end],
                topk_weights[start:end],
                w1,
                w2,
                activation,
                global_num_experts,
                local_num_experts,
                local_expert_offset,
                topk,
            )

        return unfinalized


class TrtLlmMxfp4ExpertsModular(TrtLlmMxfp4ExpertsModularBase):
    """Modular MXFP4 TRTLLM experts with unquantized activations."""

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class TrtLlmMxfp4Mxfp8ExpertsModular(TrtLlmMxfp4ExpertsModularBase):
    """Modular MXFP4 TRTLLM experts with dynamic MXFP8 activations."""

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A
