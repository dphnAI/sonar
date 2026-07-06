# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion integration for Aphrodite."""

import aphrodite.kernels.helion.ops  # noqa: F401  Auto-register all Helion ops
from aphrodite.kernels.helion.case_key import CaseKey
from aphrodite.kernels.helion.config_manager import (
    ConfigManager,
    ConfigSet,
)
from aphrodite.kernels.helion.register import (
    ConfigPicker,
    ConfiguredHelionKernel,
    HelionKernelWrapper,
    aphrodite_helion_lib,
    get_kernel_by_name,
    get_registered_kernels,
    register_kernel,
)
from aphrodite.kernels.helion.utils import canonicalize_gpu_name, get_canonical_gpu_name

__all__ = [
    # Config management
    "CaseKey",
    "ConfigManager",
    "ConfigSet",
    # Kernel registration
    "ConfigPicker",
    "ConfiguredHelionKernel",
    "HelionKernelWrapper",
    "get_kernel_by_name",
    "get_registered_kernels",
    "register_kernel",
    "aphrodite_helion_lib",
    # Utilities
    "canonicalize_gpu_name",
    "get_canonical_gpu_name",
]
