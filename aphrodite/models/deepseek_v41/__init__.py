# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

from importlib import import_module
from typing import Any

import aphrodite.envs as envs
from aphrodite.platforms import current_platform

from .quant_config import DeepseekV4FP8Config


def __getattr__(name: str) -> Any:
    if name not in {"DSparkDeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if (
        envs.APHRODITE_DSV41_SM120_NATIVE
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    ):
        backend = "sm120"
    else:
        backend = "amd" if current_platform.is_rocm() else "nvidia"
    module = "dspark" if name == "DSparkDeepseekV4ForCausalLM" else "vl_model"
    return getattr(import_module(f"{__name__}.{backend}.{module}"), name)


__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]
