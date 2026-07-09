# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Concrete fusers for the Transformers modeling backend."""

from aphrodite.model_executor.models.transformers.fusers.base import BaseFuser, StackedFuser
from aphrodite.model_executor.models.transformers.fusers.glu import GLUFuser
from aphrodite.model_executor.models.transformers.fusers.moe import MoEBlockFuser
from aphrodite.model_executor.models.transformers.fusers.qkv import QKVFuser
from aphrodite.model_executor.models.transformers.fusers.rms_norm import RMSNormFuser

__all__ = [
    "BaseFuser",
    "StackedFuser",
    "GLUFuser",
    "MoEBlockFuser",
    "QKVFuser",
    "RMSNormFuser",
]
