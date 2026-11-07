# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from aphrodite.lora.layers.base import BaseLayerWithLoRA
from aphrodite.lora.layers.classifier import (
    BaseClassificationHeadWithLoRA,
    ClassificationHeadWithLoRA,
    TokenClassificationHeadWithLoRA,
)
from aphrodite.lora.layers.column_parallel_linear import (
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
)
from aphrodite.lora.layers.custom_layer import (
    Step5FusedQKVIndexerWithLoRA,
    Step5SparseIndexerWWithLoRA,
)
from aphrodite.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from aphrodite.lora.layers.logits_processor import LogitsProcessorWithLoRA
from aphrodite.lora.layers.replicated_linear import ReplicatedLinearWithLoRA
from aphrodite.lora.layers.row_parallel_linear import (
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
)
from aphrodite.lora.layers.utils import LoRAMapping, LoRAMappingType
from aphrodite.lora.layers.vocab_parallel_embedding import VocabParallelEmbeddingWithLoRA

__all__ = [
    "BaseLayerWithLoRA",
    "Step5FusedQKVIndexerWithLoRA",
    "Step5SparseIndexerWWithLoRA",
    "VocabParallelEmbeddingWithLoRA",
    "LogitsProcessorWithLoRA",
    "ColumnParallelLinearWithLoRA",
    "ColumnParallelLinearWithShardedLoRA",
    "MergedColumnParallelLinearWithLoRA",
    "MergedColumnParallelLinearWithShardedLoRA",
    "MergedColumnParallelLinearVariableSliceWithLoRA",
    "MergedQKVParallelLinearWithLoRA",
    "MergedQKVParallelLinearWithShardedLoRA",
    "QKVParallelLinearWithLoRA",
    "QKVParallelLinearWithShardedLoRA",
    "RowParallelLinearWithLoRA",
    "RowParallelLinearWithShardedLoRA",
    "ReplicatedLinearWithLoRA",
    "LoRAMapping",
    "LoRAMappingType",
    "FusedMoEWithLoRA",
    "FusedMoE3DWithLoRA",
    "ClassificationHeadWithLoRA",
    "BaseClassificationHeadWithLoRA",
    "TokenClassificationHeadWithLoRA",
]
