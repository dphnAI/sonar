# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product

import torch

import aphrodite.ir.ops
from aphrodite.config import AphroditeConfig

from ..aphrodite_inductor_pass import (
    AphroditeFusionPatternMatcherPass,
    AphroditePatternReplacement,
)


class AddRMSNormPattern(AphroditePatternReplacement):
    def __init__(self, epsilon: float, residual_first: bool) -> None:
        self.epsilon = epsilon
        self.residual_first = residual_first

    @property
    def pattern(self):
        def _pattern(
            branch: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual_out = residual + branch if self.residual_first else branch + residual
            rms = aphrodite.ir.ops.rms_norm(residual_out, weight, self.epsilon)
            return rms, residual_out

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            branch: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return aphrodite.ir.ops.fused_add_rms_norm(branch, residual, weight, self.epsilon)

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_bf16(5, 16),
            self.empty_bf16(5, 16),
            self.empty_bf16(16),
        ]


class RMSNormReshapePattern(AphroditePatternReplacement):
    """Move a prefix-flatten before RMSNorm."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    @property
    def pattern(self):
        def _pattern(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            rms = aphrodite.ir.ops.rms_norm(input, weight, self.epsilon)
            return rms.reshape(-1, rms.shape[-1])

        return _pattern

    @property
    def replacement(self):
        def _replacement(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            input = input.reshape(-1, input.shape[-1])
            return aphrodite.ir.ops.rms_norm(input, weight, self.epsilon)

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty_bf16(1, 5, 16), self.empty_bf16(16)]


class FusedAddRMSNormReshapePattern(AphroditePatternReplacement):
    """Move a prefix-flatten before fused add RMSNorm."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rms, residual_out = aphrodite.ir.ops.fused_add_rms_norm(input, residual, weight, self.epsilon)
            return rms.reshape(-1, rms.shape[-1]), residual_out

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            original_shape = input.shape
            hidden_size = input.shape[-1]
            rms, residual_out = aphrodite.ir.ops.fused_add_rms_norm(
                input.reshape(-1, hidden_size),
                residual.reshape(-1, hidden_size),
                weight,
                self.epsilon,
            )
            return rms, residual_out.reshape(original_shape)

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_bf16(1, 5, 16),
            self.empty_bf16(1, 5, 16),
            self.empty_bf16(16),
        ]


class AddRMSNormFusionPass(AphroditeFusionPatternMatcherPass):
    """Fuse residual Add and RMSNorm emitted by the Transformers backend."""

    def __init__(self, config: AphroditeConfig) -> None:
        super().__init__(config, "add_rmsnorm_fusion_pass")
        for epsilon, residual_first in product([1e-5, 1e-6], [True, False]):
            self.register(AddRMSNormPattern(epsilon, residual_first))
        self.dump_patterns(config, self.pm_pass)


class RMSNormReshapeFusionPass(AphroditeFusionPatternMatcherPass):
    """Move post-RMSNorm flattening before the norm for downstream 2D fusions."""

    def __init__(self, config: AphroditeConfig) -> None:
        super().__init__(config, "rmsnorm_reshape_fusion_pass")
        for epsilon in [1e-5, 1e-6]:
            self.register(FusedAddRMSNormReshapePattern(epsilon))
            self.register(RMSNormReshapePattern(epsilon))
        self.dump_patterns(config, self.pm_pass)
