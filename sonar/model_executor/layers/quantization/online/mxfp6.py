# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online OCP MXFP6 quantization for Blackwell SM110."""

from typing import Literal

import torch
from torch.nn import Module

from aphrodite.config.quantization import QuantSpec
from aphrodite.logger import init_logger
from aphrodite.model_executor.kernels.linear.mxfp6 import (
    CutedslMxfp6LinearKernel,
    Mxfp6LinearLayerConfig,
)
from aphrodite.model_executor.layers.linear import UnquantizedLinearMethod
from aphrodite.model_executor.layers.quantization.online.fp8 import (
    OnlineLinearBase,
)
from aphrodite.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from aphrodite.model_executor.layers.quantization.utils.mxfp6_online_utils import (
    quantize_mxfp6_cuda,
)
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from aphrodite.model_executor.utils import replace_parameter

logger = init_logger(__name__)


def _swizzle_mxfp6_expert_scales(scales: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """Vectorized F8_128x4 scale swizzle for ``[expert, M, K/32]``."""
    experts = scales.shape[0]
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (k + 127) // 128
    padded = torch.zeros(
        (experts, num_m_tiles * 128, num_k_tiles * 4),
        dtype=scales.dtype,
        device=scales.device,
    )
    padded[:, :m, : k // 32] = scales
    return padded.view(experts, num_m_tiles, 4, 32, num_k_tiles, 4).transpose(2, 4).contiguous().view(experts, -1)


class Mxfp6OnlineLinearMethod(OnlineLinearBase):
    def __init__(self, spec: QuantSpec):
        super().__init__()
        self.weight_quant_key = spec.weight
        self.activation_quant_key = spec.activation
        if spec.weight in (kMxfp6E2M3Static, kMxfp6E2M3Dynamic):
            weight_format: Literal["e2m3", "e3m2"] = "e2m3"
        elif spec.weight in (kMxfp6E3M2Static, kMxfp6E3M2Dynamic):
            weight_format = "e3m2"
        else:
            raise ValueError(f"unsupported MXFP6 weight format: {spec.weight}")

        if spec.activation == kMxfp8Dynamic:
            activation_format: Literal["mxfp8", "mxfp6_e2m3", "mxfp6_e3m2"] = "mxfp8"
        elif spec.activation == kMxfp6E2M3Dynamic:
            activation_format = "mxfp6_e2m3"
        elif spec.activation == kMxfp6E3M2Dynamic:
            activation_format = "mxfp6_e3m2"
        else:
            raise ValueError("MXFP6 requires dynamic MXFP8 or MXFP6 activations")
        self.kernel = CutedslMxfp6LinearKernel(Mxfp6LinearLayerConfig(weight_format, activation_format))
        self.unquantized = UnquantizedLinearMethod()
        self.fallback_reason: str | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        n = sum(output_partition_sizes)
        k = input_size_per_partition
        supported, reason = self.kernel.is_supported()
        if supported:
            supported, reason = self.kernel.can_implement_shape(n, k)
        if not supported:
            self.fallback_reason = reason
            logger.warning_once(
                "Keeping an MXFP6-targeted linear layer in %s because %s",
                params_dtype,
                reason,
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if self.fallback_reason is not None:
            weight = self.get_weight_for_quantization(layer)
            replace_parameter(layer, "weight", weight.data)
            self.release_requantization_source_weights(layer)
            layer._already_called_process_weights_after_loading = True
            return

        logger.info_once("Converting full-precision weights to MXFP6 one layer at a time")
        weight = self.get_weight_for_quantization(layer)
        n, k = weight.shape
        packed, scales = quantize_mxfp6_cuda(
            weight.contiguous(),
            self.kernel.config.weight_format,
        )
        layer.mxfp6_logical_n = n
        layer.mxfp6_logical_k = k
        replace_parameter(layer, "weight", packed.data)
        replace_parameter(layer, "weight_scale", scales.data)
        self.kernel.process_weights_after_loading(layer)
        self.release_requantization_source_weights(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fallback_reason is not None:
            return self.unquantized.apply(layer, x, bias)
        return self.kernel.apply_weights(layer, x, bias)


class Mxfp6OnlineMoEMethod(OnlineMoEMethodBase):
    """Quantize BF16 MoE weights to packed MXFP6 while each layer loads."""

    @classmethod
    def can_implement(cls, moe) -> tuple[bool, str | None]:
        from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation

        if moe.activation != MoEActivation.SILU:
            return False, f"activation {moe.activation} is not supported"
        if moe.moe_parallel_config.ep_size != 1:
            return False, "expert parallel execution is not supported yet"
        if moe.has_bias:
            return False, "biased experts are not supported yet"
        if moe.swiglu_limit is not None:
            return False, "SwiGLU clamping is not supported yet"
        if (
            moe.hidden_dim < 128
            or moe.hidden_dim % 128
            or moe.intermediate_size_per_partition < 128
            or moe.intermediate_size_per_partition % 128
        ):
            return False, "hidden and intermediate dimensions must be multiples of 128"
        return True, None

    def __init__(self, *, layer: torch.nn.Module, spec: QuantSpec):
        super().__init__(layer.moe_config)
        self.weight_quant_key = spec.weight
        self.activation_quant_key = spec.activation
        if spec.weight == kMxfp6E2M3Static:
            weight_format: Literal["e2m3", "e3m2"] = "e2m3"
        elif spec.weight == kMxfp6E3M2Static:
            weight_format = "e3m2"
        else:
            raise ValueError(f"unsupported MXFP6 MoE weight format: {spec.weight}")
        if spec.activation == kMxfp8Dynamic:
            activation_format: Literal["mxfp8", "mxfp6_e2m3", "mxfp6_e3m2"] = "mxfp8"
        elif spec.activation == kMxfp6E2M3Dynamic:
            activation_format = "mxfp6_e2m3"
        elif spec.activation == kMxfp6E3M2Dynamic:
            activation_format = "mxfp6_e3m2"
        else:
            raise ValueError("MXFP6 MoE requires dynamic MXFP8 or MXFP6 activations")
        self.weight_format = weight_format
        self.activation_format = activation_format

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % 128 or intermediate_size_per_partition % 128:
            raise ValueError("native MXFP6 MoE dimensions must be divisible by 128")
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        del layer
        return getattr(self, "moe_quant_config", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        from aphrodite.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from aphrodite.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        from aphrodite.model_executor.layers.fused_moe.experts.cutedsl_mxfp6_moe import (
            CutedslMxfp6Experts,
        )

        logger.info_once("Converting full-precision weights to MXFP6 one layer at a time")
        w13_weight, w2_weight = self.get_weights_for_quantization(layer)
        self._zero_padding(layer, w13_weight, w2_weight)
        w13_k = w13_weight.shape[2]
        w13, s13 = quantize_mxfp6_cuda(w13_weight.contiguous(), self.weight_format)
        s13 = _swizzle_mxfp6_expert_scales(s13, w13.shape[1], w13_k)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", s13)
        del w13_weight

        # Release the full-precision gate/up tensor before converting the down
        # projection. This bounds conversion memory to one expert tensor.
        w2_k = w2_weight.shape[2]
        w2, s2 = quantize_mxfp6_cuda(w2_weight.contiguous(), self.weight_format)
        s2 = _swizzle_mxfp6_expert_scales(s2, w2.shape[1], w2_k)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", s2)
        del w2_weight
        self.release_requantization_source_weights(layer)

        act_dtype = self.activation_format
        weight_dtype = f"mxfp6_{self.weight_format}"
        quant_config = FusedMoEQuantConfig.make(
            act_dtype,
            weight_dtype=weight_dtype,
            block_shape=[1, 32],
            w1_scale=s13,
            w2_scale=s2,
            is_scale_swizzled=True,
        )
        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
        )
        assert prepare_finalize is not None
        experts = CutedslMxfp6Experts(
            moe_config=self.moe,
            quant_config=quant_config,
            weight_format=self.weight_format,
            activation_format=self.activation_format,
        )
        import aphrodite.model_executor.layers.fused_moe.modular_kernel as mk

        self.moe_quant_config = quant_config
        self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
        layer._already_called_process_weights_after_loading = True
