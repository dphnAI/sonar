# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM120 text decoding with the existing Sonar vision implementation."""

from collections.abc import Iterable

import torch
from torch import nn

from aphrodite.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from aphrodite.model_executor.models.vision import is_vit_use_data_parallel
from aphrodite.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
)
from aphrodite.models.deepseek_v41.common.mm_preprocess import (
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from aphrodite.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV41ForCausalLM as UpstreamDeepseekV41ForCausalLM,
)
from aphrodite.models.deepseek_v41.nvidia.vl_model import (
    _make_deepseek_v4_vl_weights_mapper,
)
from aphrodite.multimodal import MULTIMODAL_REGISTRY

from .model import DeepseekV41LLMForCausalLM


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV41ForCausalLM(UpstreamDeepseekV41ForCausalLM):
    """Multimodal entry point for DeepSeek-V4.1 checkpoints with a vision tower.

    ``SupportsEagle3`` (aux hidden-state plumbing for MTP/DSpark drafters)
    delegates through ``language_model`` via the protocol defaults.
    """

    supports_encoder_tp_data = True

    # The MoE router applies bias_vl to raw image-span token IDs.
    requires_raw_input_tokens = True

    def __init__(self, *, aphrodite_config, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = aphrodite_config.model_config
        config = model_config.hf_config
        self.config = config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        # The tower is always built; _mark_tower_model stubs it out
        # (StageMissingLayer, weights skipped) when the image limit is 0.
        with self._mark_tower_model(aphrodite_config, {"image"}):
            self.use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
            self.vision = DeepseekV4ViT(config)
            self.aligner = DeepseekV4Aligner(config)
            self.image_start = nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32))

            self.vision.to(dtype=model_config.dtype)
            self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(aphrodite_config):
            self.language_model = DeepseekV41LLMForCausalLM(
                aphrodite_config=aphrodite_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # The outer mapper (see load_weights) fully resolves HF names into
        # this wrapper's namespace before AutoWeightsLoader strips the
        # "language_model." prefix and delegates to the child's load_weights,
        # so the child's own mapper must be a no-op. Its suffix rules are not
        # idempotent (e.g. "lm_head.weight".endswith("head.weight") would
        # re-fire "head.weight" -> "lm_head.weight").
        self.language_model.hf_to_aphrodite_mapper = WeightsMapper()
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_aphrodite_mapper = _make_deepseek_v4_vl_weights_mapper(expert_dtype, "weight_scale_inv")

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    supports_b12x_native_preparation = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Prefix groups may recur across checkpoint files. Consume each tensor
        # immediately; sorting payloads retains the entire checkpoint on device.
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_aphrodite_mapper)

    def process_weights_after_loading(self) -> None:
        self.language_model.process_weights_after_loading()
