# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from aphrodite.config import AphroditeConfig
from aphrodite.model_executor.layers.attention import CrossAttention
from aphrodite.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_model_state(
    aphrodite_config: AphroditeConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    # Let the model provide its own ModelState if it defines one.
    if hasattr(model, "get_model_state_cls"):
        cls = model.get_model_state_cls()
        return cls(aphrodite_config, model, encoder_cache, device)

    # Cross-attention encoder-decoder models (Whisper, CohereASR, NemotronParse, ...)
    if any(isinstance(m, CrossAttention) for m in model.modules()):
        from aphrodite.v1.worker.gpu.model_states.encoder_decoder import (
            EncoderDecoderModelState,
        )

        return EncoderDecoderModelState(aphrodite_config, model, encoder_cache, device)

    if aphrodite_config.model_config.is_hybrid:
        from aphrodite.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState(aphrodite_config, model, encoder_cache, device)

    from aphrodite.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(aphrodite_config, model, encoder_cache, device)
