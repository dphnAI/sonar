# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

import torch

from aphrodite.config import AphroditeConfig, get_layers_from_aphrodite_config
from aphrodite.distributed import get_dcp_group

if TYPE_CHECKING:
    from aphrodite.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


def check_attention_cp_compatibility(aphrodite_config: AphroditeConfig) -> None:
    pcp_size = aphrodite_config.parallel_config.prefill_context_parallel_size
    dcp_size = aphrodite_config.parallel_config.decode_context_parallel_size
    interleave_size = aphrodite_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_aphrodite_config(aphrodite_config, layer_type)
        for layer in layers.values():
            get_attn_backend = getattr(layer, "get_attn_backend", None)
            if pcp_size > 1 and get_attn_backend is not None:
                backend = get_attn_backend()
                assert backend.supports_pcp(), (
                    f"PCP requires attention backend support, but {backend.get_name()} does not support PCP."
                )
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if aphrodite_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    f"MTP with cp_kv_cache_interleave_size > 1 is not supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )


def get_kv_cache_shard_count() -> int:
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size


def get_total_cp_world_size() -> int:
    return get_kv_cache_shard_count()


def should_skip_dcp_context_attention(context_kv_lens: torch.Tensor) -> bool:
    return bool(context_kv_lens.max().item() == 0)
