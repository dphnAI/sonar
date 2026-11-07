# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Concrete fusers for the Transformers modeling backend."""

from sonar.model_executor.models.transformers.fusers.attention import AttentionFuser
from sonar.model_executor.models.transformers.fusers.base import (
    BaseFuser,
    RewriteFuser,
    StackedFuser,
)
from sonar.model_executor.models.transformers.fusers.glu import GLUFuser
from sonar.model_executor.models.transformers.fusers.merged_column import (
    MergedColumnParallelFuser,
)
from sonar.model_executor.models.transformers.fusers.mla import MLAFuser
from sonar.model_executor.models.transformers.fusers.moe import MoEBlockFuser
from sonar.model_executor.models.transformers.fusers.packed_qkv import PackedQKVFuser
from sonar.model_executor.models.transformers.fusers.qkv import QKVFuser
from sonar.model_executor.models.transformers.fusers.rms_norm import RMSNormFuser

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
