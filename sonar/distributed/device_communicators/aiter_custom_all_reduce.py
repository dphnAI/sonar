# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sonar-owned wrapper over AITER's `CustomAllreduce`.

`CudaCommunicator` stores one of these as `aiter_ar_comm` when
`SONAR_ROCM_USE_AITER_CUSTOM_AR` is set, so plain all-reduce and the
fused all-reduce + RMSNorm path share a single AITER instance with its IPC
buffers.
"""

import torch
from torch.distributed import ProcessGroup

from sonar.logger import init_logger

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
        *,
        register_graph_buffers: bool = True,
    ):
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as _AiterCustomAllreduce,
        )

        if max_size is None:
            max_size = self.MAX_SIZE

        kwargs = {}
        if not register_graph_buffers:
            kwargs["enable_register_for_capturing"] = False
        self._impl = _AiterCustomAllreduce(group, device, max_size=max_size, **kwargs)
        self._register_graph_buffers = register_graph_buffers

    @property
    def aiter_ca(self):
        return self._impl

    @property
    def disabled(self) -> bool:
        return self._impl.disabled

    @property
    def registers_graph_buffers(self) -> bool:
        """Whether graph capture registers its inputs once expandable segments
        are off, which AITER decides when capture begins from v0.1.26."""
        from sonar.platforms.rocm import on_gfx1250

        return (
            self._register_graph_buffers
            and not self.disabled
            and not on_gfx1250()
            and getattr(type(self._impl), "defers_capture_registration", False)
        )

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return self._impl.should_custom_ar(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        return self._impl.custom_all_reduce(inp)

    def should_custom_ag(self, inp: torch.Tensor) -> bool:
        return self._impl.should_custom_ag(inp)

    def should_custom_rs(self, inp: torch.Tensor, dim: int) -> bool:
        return self._impl.should_custom_rs(inp, dim)

    def custom_all_gather(self, inp: torch.Tensor, dim: int = 0) -> torch.Tensor | None:
        return self._impl.custom_all_gather(inp, dim=dim)

    def all_gather_unreg(self, inp: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return self._impl.all_gather_unreg(inp, dim=dim)

    def custom_reduce_scatter(self, inp: torch.Tensor, out: torch.Tensor, dim: int = 0) -> torch.Tensor | None:
        return self._impl.custom_reduce_scatter(inp, out, dim=dim)

    def use_1stage_fused_ar_rms(self, inp: torch.Tensor) -> bool:
        """Whether AITER's fused allreduce+RMSNorm runs as its one-stage kernel.

        Mirrors the launcher contract of aiter's `fused_allreduce_rmsnorm`
        (csrc/include/custom_all_reduce.cuh): rows of 16-byte packs, at most
        1024 packs per row, at most 80 tokens, and the byte cap of the
        one-stage custom allreduce for this TP size and topology. Outside it
        the fused op runs the two-stage variant (cross-device reduce-scatter
        + local norm), which is slower than an explicit `all_reduce` + norm,
        so callers that can fall back should require this. Capture-static:
        depends only on shape, dtype, TP size and topology.
        """
        hidden_dim = inp.shape[-1]
        # Token cap first: prefill-sized inputs leave here with one comparison.
        if inp.numel() // hidden_dim > 80:
            return False
        if inp.dtype not in (torch.bfloat16, torch.float16):
            return False
        pack_size = 16 // inp.element_size()
        if hidden_dim % pack_size != 0 or hidden_dim // pack_size > 1024:
            return False
        ca = self._impl
        world_size = ca.world_size
        if world_size == 2:
            return True
        if not ca.fully_connected:
            return False
        total_bytes = inp.numel() * inp.element_size()
        if world_size <= 4:
            return total_bytes < 256 * 1024
        if world_size <= 8:
            return total_bytes < 128 * 1024
        return False

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
