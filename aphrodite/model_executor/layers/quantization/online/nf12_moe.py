# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from aphrodite.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from aphrodite.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase
from aphrodite.model_executor.utils import replace_parameter
from aphrodite.platforms import current_platform


class Nf12OnlineMoEMethod(OnlineMoEMethodBase):
    """Losslessly pack BF16 experts and execute the packaged NF12 MoE kernels."""

    @classmethod
    def can_implement(cls, layer: RoutedExperts) -> tuple[bool, str | None]:
        moe = layer.moe_config
        if moe.in_dtype != torch.bfloat16 or layer.params_dtype != torch.bfloat16:
            return False, "NF12 requires BF16 activations and weights"
        if not current_platform.is_cuda() or not current_platform.has_device_capability(80):
            return False, "NF12 requires an NVIDIA GPU with compute capability 8.0 or newer"
        if torch.device(moe.device).type != "cuda":
            return False, "NF12 requires CUDA weights"
        if moe.activation != MoEActivation.SILU:
            return False, "NF12 supports only gated SiLU experts"
        if moe.has_bias:
            return False, "NF12 does not support expert biases"
        if any(
            value is not None
            for value in (
                moe.swiglu_limit,
                moe.swiglu_alpha,
                moe.swiglu_beta,
                layer.swiglu_limit,
                layer.swiglu_alpha,
                layer.swiglu_beta,
            )
        ):
            return False, "NF12 does not support SwiGLU clamping or scaling"
        if moe.is_lora_enabled:
            return False, "NF12 does not support MoE LoRA"
        parallel = moe.moe_parallel_config
        if parallel.enable_eplb:
            return False, "NF12 does not support EPLB or elastic expert parallelism"
        if parallel.all2all_backend != "allgather_reducescatter":
            return False, "NF12 supports only allgather_reducescatter MoE communication"
        if moe.is_fused_shared_expert_requested or layer.expert_map_manager.num_fused_shared_experts:
            return False, "NF12 requires shared experts to execute separately"
        if layer.apply_router_weight_on_input and moe.experts_per_token != 1:
            return False, "NF12 router-weight-on-input requires topk=1"
        if (
            moe.hidden_dim <= 0
            or moe.hidden_dim % 64
            or moe.intermediate_size_per_partition <= 0
            or moe.intermediate_size_per_partition % 64
        ):
            return False, "NF12 MoE hidden and intermediate dimensions must be multiples of 64"
        return True, None

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def topk_indices_dtype(self) -> torch.dtype:
        return torch.int32

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        # NF12 owns its encoded metadata; the legacy runner needs no MK config.
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        import nf12

        if layer.expert_map_manager.num_fused_shared_experts:
            raise ValueError("NF12 requires shared experts to execute separately")
        if layer.apply_router_weight_on_input and self.moe.experts_per_token != 1:
            raise ValueError("NF12 router-weight-on-input requires topk=1")

        w13, w2 = self.get_weights_for_quantization(layer)
        if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            raise ValueError("NF12 lossless encoding requires BF16 expert weights")
        self._zero_padding(layer, w13, w2)

        # Encoding loads the native library before compilation/graph capture.
        encoded = nf12.encode_moe(w13)
        replace_parameter(layer, "w13_weight", encoded.packed)
        replace_parameter(layer, "w13_escapes", encoded.escapes)
        replace_parameter(layer, "w13_escape_offsets", encoded.escape_offsets)
        replace_parameter(layer, "w13_exp_base", encoded.exp_base)
        del w13, encoded

        encoded = nf12.encode_moe(w2)
        replace_parameter(layer, "w2_weight", encoded.packed)
        replace_parameter(layer, "w2_escapes", encoded.escapes)
        replace_parameter(layer, "w2_escape_offsets", encoded.escape_offsets)
        replace_parameter(layer, "w2_exp_base", encoded.exp_base)
        del w2, encoded
        self.release_requantization_source_weights(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        import nf12

        if layer.apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise ValueError("NF12 router-weight-on-input requires topk=1")
            x = (x.float() * topk_weights).to(x.dtype)
            topk_weights = torch.ones_like(topk_weights)

        # Reconstruct metadata from live parameters, not cached tensor aliases:
        # reload and device movement may replace the registered tensors.
        w13 = nf12.NF12MoEWeight(
            layer.w13_weight,
            layer.w13_escapes,
            layer.w13_escape_offsets,
            layer.w13_exp_base,
        )
        w2 = nf12.NF12MoEWeight(
            layer.w2_weight,
            layer.w2_escapes,
            layer.w2_escape_offsets,
            layer.w2_exp_base,
        )
        # MoERunner owns shared-expert execution and cross-rank reductions.
        # The package uses the current CUDA stream and captures without decode.
        return nf12.fused_moe(x, w13, w2, topk_weights, topk_ids, expert_map=layer.expert_map)
