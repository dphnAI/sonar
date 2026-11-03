# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from aphrodite.model_executor.kernels.linear import init_mxfp4_linear_kernel
from aphrodite.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from aphrodite.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from aphrodite.model_executor.utils import move_parameter

from .inc_scheme import INCLinearScheme

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig


class INCMxfp4LinearMethod(INCLinearScheme):
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        if not isinstance(layer_config.group_size, int):
            raise ValueError(f"INC MXFP4 requires scalar group_size, but found group_size={layer_config.group_size!r}.")
        self.group_size = layer_config.group_size or 32
        self.kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        move_parameter(layer, "weight_packed", "weight")
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
