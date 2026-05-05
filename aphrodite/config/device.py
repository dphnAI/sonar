# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal

import torch
from pydantic import ConfigDict, SkipValidation

from aphrodite.config.utils import config
from aphrodite.utils.hashing import safe_hash

Device = Literal["auto", "cuda", "cpu", "tpu", "xpu", "metal"]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    """Configuration for the device to use for Aphrodite execution."""

    device: SkipValidation[Device | torch.device | None] = "auto"
    """Device type for Aphrodite execution.
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
    on the current platform."""
    device_type: str = field(init=False)
    """Device type from the current platform. This is set in
    `__post_init__`."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # the device/platform information will be summarized
        # by torch/aphrodite automatically.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.device == "auto":
            # Automated device type detection
            from aphrodite.platforms import current_platform

            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `APHRODITE_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue."
                )
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some device types require processing inputs on CPU. Metal is an
        # MLX-backed platform, but Aphrodite tensors still flow through CPU/MPS
        # compatible paths rather than a torch "metal" device.
        if self.device_type in ["tpu"]:
            self.device = None
        elif self.device_type == "metal":
            self.device = torch.device("cpu")
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)
