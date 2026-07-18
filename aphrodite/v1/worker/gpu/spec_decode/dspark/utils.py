# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from aphrodite.config import AphroditeConfig, replace
from aphrodite.distributed.parallel_state import get_pp_group
from aphrodite.model_executor.model_loader import get_model
from aphrodite.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dspark_model(target_model: nn.Module, aphrodite_config: AphroditeConfig) -> nn.Module:
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from aphrodite.compilation.backends import set_model_tag
    from aphrodite.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    draft_aphrodite_config = replace(
        aphrodite_config,
        attention_config=replace(
            aphrodite_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                aphrodite_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else aphrodite_config.cache_config
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(aphrodite_config=draft_aphrodite_config, model_config=draft_model_config)

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    target_language_model = (
        target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(draft_model, "has_own_embed_tokens", draft_embed, target_embed):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
