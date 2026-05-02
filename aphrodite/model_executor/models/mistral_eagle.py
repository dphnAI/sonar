# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.config import AphroditeConfig
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.layernorm import RMSNorm
from aphrodite.model_executor.layers.linear import RowParallelLinear
from aphrodite.model_executor.layers.logits_processor import LogitsProcessor
from aphrodite.model_executor.layers.quantization.base_config import QuantizationConfig
from aphrodite.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from aphrodite.model_executor.models.interfaces import MultiModalEmbeddings
from aphrodite.model_executor.models.llama import LlamaConfig
from aphrodite.model_executor.models.mistral import (
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralModel,
)
from aphrodite.model_executor.models.utils import (
    _merge_multimodal_embeddings,
    get_draft_quant_config,
    maybe_prefix,
)

logger = init_logger(__name__)


class EagleMistralDecoderLayer(MistralDecoderLayer):
    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        super().__init__(aphrodite_config, prefix=prefix, config=config)

    def get_quant_config(self, aphrodite_config: AphroditeConfig) -> QuantizationConfig | None:
        return get_draft_quant_config(aphrodite_config)


@support_torch_compile
class EagleMistralModel(MistralModel):
    def __init__(
        self,
        *,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        # Bypass MistralModel.__init__ to avoid creating duplicate attention
        # layer entries in the global context.
        nn.Module.__init__(self)
        self.config = aphrodite_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        # Get drafter's quantization config
        self.quant_config = get_draft_quant_config(aphrodite_config)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
            quant_config=self.quant_config,
        )

        self.layers = nn.ModuleList(
            [
                EagleMistralDecoderLayer(
                    aphrodite_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                    config=self.config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.fc = RowParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Pretend embed_tokens is loaded; the actual weight is shared
        # from the target model at runtime by `load_eagle_model`.
        return super().load_weights(weights) | {"embed_tokens.weight"}


class EagleMistralForCausalLM(MistralForCausalLM):
    mistral_mapping = MistralForCausalLM.mistral_mapping | {
        "eagle_linear": "model.fc",
    }

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "") -> None:
        # Bypass MistralForCausalLM.__init__ to use the draft model config
        # and to avoid creating an lm_head.
        nn.Module.__init__(self)
        self.config = aphrodite_config.speculative_config.draft_model_config.hf_config
        target_layer_num = aphrodite_config.model_config.get_num_layers(aphrodite_config.parallel_config)
        self.model = EagleMistralModel(
            aphrodite_config=aphrodite_config, prefix="model", start_layer_id=target_layer_num
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size, scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = super().embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        assert is_multimodal is not None

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
