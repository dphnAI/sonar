# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aphrodite-owned wrapper over AITER's ``CustomAllreduce``.

``CudaCommunicator`` stores one of these as ``aiter_ar_comm`` when
``APHRODITE_ROCM_USE_AITER_CUSTOM_AR`` is set, so plain all-reduce and the
fused all-reduce + RMSNorm path share a single AITER instance with its IPC
buffers.
"""

import torch
from torch.distributed import ProcessGroup

from aphrodite.logger import init_logger

logger = init_logger(__name__)


class AiterCustomAllreduce:
    # Default IPC buffer size for AITER's CustomAllreduce.
    MAX_SIZE: int = 8192 * 1024 * 8 * 2

    @classmethod
    def effective_max_size(cls) -> int:
        """Max input byte size eligible for AITER custom all-reduce."""
        return cls.MAX_SIZE // 2

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int | None = None,
    ):
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as _AiterCustomAllreduce,
        )

        if max_size is None:
            max_size = self.MAX_SIZE

        self._impl = _AiterCustomAllreduce(group, device, max_size=max_size)

    @property
    def aiter_ca(self):
        return self._impl

    @property
    def disabled(self) -> bool:
        return self._impl.disabled

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return self._impl.should_custom_ar(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        return self._impl.custom_all_reduce(inp)

    def capture(self):
        return self._impl.capture()

    def close(self) -> None:
        self._impl.close()

    @property
    def supports_dynamic_hidden_dim(self) -> bool:
        """Whether AITER's fused AR+RMS launcher accepts runtime hidden_dim."""
        return hasattr(self._impl, "_pool")

    @staticmethod
    def build_supports_per_group_quant() -> bool:
        """True if the running AITER build exposes AR+RMS+per-group quant."""
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as _AiterCustomAllreduce,
        )

        return hasattr(_AiterCustomAllreduce, "fused_ar_rms_per_group_quant")

    @property
    def supports_per_group_quant(self) -> bool:
        return self.build_supports_per_group_quant()
