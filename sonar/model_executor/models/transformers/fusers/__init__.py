# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Concrete fusers for the Transformers modeling backend."""

from aphrodite.model_executor.models.transformers.fusers.attention import AttentionFuser
from aphrodite.model_executor.models.transformers.fusers.base import (
    BaseFuser,
    RewriteFuser,
    StackedFuser,
)
from aphrodite.model_executor.models.transformers.fusers.glu import GLUFuser
from aphrodite.model_executor.models.transformers.fusers.merged_column import (
    MergedColumnParallelFuser,
)
from aphrodite.model_executor.models.transformers.fusers.mla import MLAFuser
from aphrodite.model_executor.models.transformers.fusers.moe import MoEBlockFuser
from aphrodite.model_executor.models.transformers.fusers.packed_qkv import PackedQKVFuser
from aphrodite.model_executor.models.transformers.fusers.qkv import QKVFuser
from aphrodite.model_executor.models.transformers.fusers.rms_norm import RMSNormFuser

__all__ = [
    "AttentionFuser",
    "BaseFuser",
    "RewriteFuser",
    "StackedFuser",
    "GLUFuser",
    "MergedColumnParallelFuser",
    "MLAFuser",
    "MoEBlockFuser",
    "PackedQKVFuser",
    "QKVFuser",
    "RMSNormFuser",
]
