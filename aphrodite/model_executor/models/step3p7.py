# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jurassic model."""

import torch

from aphrodite.config import AphroditeConfig
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.activation import get_act_fn
from aphrodite.model_executor.layers.linear import ColumnParallelLinear

from .step3_vl import Step3VLForConditionalGeneration
from .step_vl import PerceptionEncoder
from .utils import WeightsMapper, init_aphrodite_registered_model, maybe_prefix
from .vision import run_dp_sharded_vision_model

logger = init_logger(__name__)


class Step3p7ForConditionalGeneration(Step3VLForConditionalGeneration):
    hf_to_aphrodite_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_model.": "vision_model.",
            "model.vit_large_projector.": "vit_large_projector.",
            "model.vit_large_projector": "vit_large_projector",
            "model.language_model.": "language_model.model.",
            "model.language_model": "language_model.model",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "lm_head": "language_model.lm_head",
        },
        orig_to_new_substr={
            ".attn.in_proj_weight": ".attn.qkv_proj.weight",
            ".attn.in_proj_bias": ".attn.qkv_proj.bias",
            ".mlp.c_fc": ".mlp.fc1",
            ".mlp.c_proj": ".mlp.fc2",
        },
    )

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "") -> None:
        super(Step3VLForConditionalGeneration, self).__init__()

        config = aphrodite_config.model_config.hf_config
        multimodal_config = aphrodite_config.model_config.multimodal_config
        quant_config = aphrodite_config.quant_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        with self._mark_tower_model(aphrodite_config, "image"):
            self.vision_model = PerceptionEncoder(
                config.vision_config,
                get_act_fn(config.vision_config.hidden_act),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.vit_large_projector = ColumnParallelLinear(
                config.vision_config.width * 4,
                config.text_config.hidden_size,
                bias=config.projector_bias,
                gather_output=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vit_large_projector"),
                disable_tp=self.use_data_parallel,
            )

        with self._mark_language_model(aphrodite_config):
            self.language_model = init_aphrodite_registered_model(
                aphrodite_config=aphrodite_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _get_vision_model_output(self, input_tensor: torch.Tensor | None) -> torch.Tensor | None:
        if input_tensor is None:
            return None
        if self.use_data_parallel:
            return run_dp_sharded_vision_model(input_tensor, self.vision_model)
        return self.vision_model(input_tensor)

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features, _ = self.vit_large_projector(image_features)
        return image_features
