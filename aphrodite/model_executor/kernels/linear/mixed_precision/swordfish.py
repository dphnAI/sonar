# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwordfishLinearKernel: w4a16 GEMM for the Blackwell sm100 family
(datacenter sm100 + Thor sm110). GPTQ u4b8 only in v1."""

import torch

from aphrodite import _custom_ops as ops
from aphrodite.model_executor.layers.quantization.utils.swordfish_utils import (
    check_swordfish_supports_shape,
    query_swordfish_supported_group_sizes,
    query_swordfish_supported_quant_types,
)
from aphrodite.model_executor.parameter import (
    BaseAphroditeParameter,
    permute_param_layout_,
)
from aphrodite.platforms import current_platform

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class SwordfishLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Swordfish only supported on CUDA"

        capability = current_platform.get_device_capability()
        # sm100 family only, datacenter 10.x and Thor 11.x. Consumer
        # Blackwell (12.x) is a different SM and is untested.
        if capability is None or capability.major not in (10, 11):
            return (
                False,
                "Swordfish requires the sm100 family (compute capability "
                f"10.x or 11.x), got {capability}",
            )

        if c.has_g_idx:
            return False, "Act reordering (g_idx) not supported by Swordfish v1"

        if c.zero_points:
            return False, "Zero points not supported by Swordfish v1"

        supported_types = query_swordfish_supported_quant_types(c.zero_points)
        if c.weight_type not in supported_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by Swordfish v1, "
                f"supported: {supported_types}",
            )

        if c.group_size not in query_swordfish_supported_group_sizes(c.act_type):
            return (
                False,
                f"Group size ({c.group_size}) / act type ({c.act_type}) not "
                "supported by Swordfish v1",
            )

        return check_swordfish_supports_shape(
            c.partition_weight_shape[0], c.partition_weight_shape[1]
        )

    # weight_packed has {input_dim 0, output_dim 1, packed_dim 0} and
    # weight_scale has {input_dim 0, output_dim 1}.
    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        size_k, size_n = c.partition_weight_shape

        def transform_w_q(x):
            assert isinstance(x, BaseAphroditeParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.swordfish_prepack_B(x.data.contiguous(), size_k, size_n)
            return x

        def transform_w_s(x):
            assert isinstance(x, BaseAphroditeParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        # The decode/prefill crossover lives inside the C++ op. A Python
        # branch would be baked in at torch.compile trace time.
        output = ops.swordfish_mm(
            x_2d,
            w_q,
            w_s,
            c.group_size,
            c.partition_weight_shape[0],
            c.partition_weight_shape[1],
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
