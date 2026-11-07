# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clef-Flash: the Qwen3.5 backbone with a joint schema decision head."""

import torch

from aphrodite.config import AphroditeConfig
from aphrodite.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from aphrodite.model_executor.model_loader.default_loader import DefaultModelLoader
from aphrodite.multimodal import MULTIMODAL_REGISTRY

from .clef_joint_schema import JOINT_HEAD_WEIGHTS, JointSchemaHead, JointSchemaPooler
from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ClefFlashForJointSchema(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__(aphrodite_config=aphrodite_config, prefix=prefix)
        model_config = aphrodite_config.model_config
        self.pooler = JointSchemaPooler(
            JointSchemaHead(**model_config.hf_config.joint_head_config),
            self._embed_lexical,
        )
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="pooler.head.",
                allow_patterns_overrides=[JOINT_HEAD_WEIGHTS],
            )
        ]

    def _embed_lexical(self, token_ids: torch.Tensor) -> torch.Tensor:
        return VocabParallelEmbedding.forward(self.language_model.lm_head, token_ids)
