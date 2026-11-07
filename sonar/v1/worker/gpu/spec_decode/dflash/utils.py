# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from sonar.config import SonarConfig, replace
from sonar.config.speculative import resolve_draft_kv_cache_dtype
from sonar.logger import init_logger
from sonar.model_executor.model_loader import get_model
from sonar.model_executor.model_loader.utils import get_draft_load_config
from sonar.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from sonar.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)

_DCP_KV_REPLICATION_ARCHS = {"DFlashDraftModel", "Qwen3DSparkModel"}


def resolve_dflash_attention_backend(sonar_config: SonarConfig):
    """Select the DCP-capable backend for a KV-head-replicated draft."""
    speculative_config = sonar_config.speculative_config
    assert speculative_config is not None
    backend = speculative_config.attention_backend
    needs_head_replication = dflash_needs_dcp_kv_head_replication(sonar_config)
    if not needs_head_replication:
        return backend

    from sonar.v1.attention.backends.registry import AttentionBackendEnum

    if backend is None:
        return AttentionBackendEnum.FLASH_ATTN
    if backend != AttentionBackendEnum.FLASH_ATTN:
        raise ValueError(
            "DFlash/DSpark with DCP-replicated draft KV heads requires "
            f"attention_backend=FLASH_ATTN, got {backend.name}."
        )
    return backend


def dflash_needs_dcp_kv_head_replication(sonar_config: SonarConfig) -> bool:
    speculative_config = sonar_config.speculative_config
    assert speculative_config is not None
    parallel_config = sonar_config.parallel_config
    dcp_size = parallel_config.decode_context_parallel_size
    if dcp_size == 1:
        return False

    draft_model_config = speculative_config.draft_model_config
    total_kv_heads = draft_model_config.get_total_num_kv_heads()
    tp_size = parallel_config.tensor_parallel_size
    needs_replication = total_kv_heads >= tp_size
    if needs_replication and not (set(draft_model_config.architectures or []) & _DCP_KV_REPLICATION_ARCHS):
        raise NotImplementedError(
            "DFlash/DSpark with sharded draft KV heads under DCP requires a "
            "draft architecture that replicates its KV cache across the DCP "
            "group, but got "
            f"{draft_model_config.architectures}."
        )
    if not needs_replication:
        total_q_heads = draft_model_config.model_arch_config.total_num_attention_heads
        kv_replicated_across_dcp = (
            tp_size > total_kv_heads
            and dcp_size <= tp_size // total_kv_heads
            and (total_q_heads // total_kv_heads) % dcp_size == 0
        )
        if not kv_replicated_across_dcp:
            raise NotImplementedError(
                "DFlash/DSpark under DCP requires draft KV heads to be fully "
                "sharded or replicated across the DCP group; got "
                f"q_heads={total_q_heads}, kv_heads={total_kv_heads}, "
                f"tp={tp_size}, dcp={dcp_size}."
            )
    return needs_replication


def resolve_dflash_cache_dtype(sonar_config: SonarConfig):
    """Use a non-quantized cache when DCP replicates draft KV heads."""
    if dflash_needs_dcp_kv_head_replication(sonar_config):
        logger.warning_once(
            "DFlash/DSpark with DCP requires replicated draft KV heads; "
            "using the draft model dtype for its KV cache. The target KV "
            "cache dtype is unchanged."
        )
        return "auto"
    speculative_config = sonar_config.speculative_config
    assert speculative_config is not None
    return resolve_draft_kv_cache_dtype(
        speculative_config,
        sonar_config.cache_config.cache_dtype,
    )


def load_dflash_model(target_model: nn.Module, sonar_config: SonarConfig) -> nn.Module:
    from sonar.compilation.backends import set_model_tag
    from sonar.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = sonar_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_sonar_config = replace(
        sonar_config,
        attention_config=replace(
            sonar_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=resolve_dflash_attention_backend(sonar_config),
        ),
        cache_config=replace(
            sonar_config.cache_config,
            cache_dtype=resolve_dflash_cache_dtype(sonar_config),
        ),
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(sonar_config)),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(sonar_config=draft_sonar_config, model_config=draft_model_config)

    target_language_model = (
        target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
