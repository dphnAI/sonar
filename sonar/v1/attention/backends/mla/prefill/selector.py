# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selector for MLA prefill backends.

This module provides functions for selecting the appropriate MLA prefill
backend based on device capabilities and configuration.
"""

from functools import cache
from typing import TYPE_CHECKING, NamedTuple

import torch

from aphrodite.logger import init_logger
from aphrodite.v1.attention.backends.mla.prefill.base import MLADimensions

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig
    from aphrodite.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

logger = init_logger(__name__)


class MLAPrefillSelectorConfig(NamedTuple):
    """Hashable configuration for MLA prefill backend selection.

    This is analogous to AttentionSelectorConfig and contains model-specific
    configuration needed to select an MLA prefill backend, extracted from
    AphroditeConfig into a hashable form for caching.
    """

    dtype: torch.dtype
    mla_dimensions: MLADimensions = MLADimensions(
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
    )

    def __repr__(self):
        return f"MLAPrefillSelectorConfig(dtype={self.dtype}, mla_dimensions={self.mla_dimensions})"


def get_mla_prefill_backend(
    aphrodite_config: "AphroditeConfig | None",
) -> "type[MLAPrefillBackend]":
    """Select the MLA prefill backend based on configuration and device.

    This function first checks for explicit user preferences via
    mla_prefill_backend in AttentionConfig, then falls back to automatic
    priority-based selection.

    Args:
        aphrodite_config: The Aphrodite configuration. May be None before model
            resolution, in which case defaults are used.

    Returns:
        The selected prefill backend class.

    """
    from aphrodite.platforms import current_platform

    model_config = aphrodite_config.model_config if aphrodite_config is not None else None
    if model_config is None:
        selector_config = MLAPrefillSelectorConfig(dtype=torch.get_default_dtype())
    else:
        hf_text_config = model_config.hf_text_config
        selector_config = MLAPrefillSelectorConfig(
            dtype=model_config.dtype,
            mla_dimensions=MLADimensions(
                qk_nope_head_dim=getattr(hf_text_config, "qk_nope_head_dim", 0),
                qk_rope_head_dim=getattr(hf_text_config, "qk_rope_head_dim", 0),
                v_head_dim=getattr(hf_text_config, "v_head_dim", 0),
            ),
        )

    if current_platform.is_cpu():
        # CPUs have a single fixed MLA prefill backend, so explicit backend
        # configuration does not apply; always auto-select via the platform.
        return _auto_select_mla_prefill_backend(selector_config)

    attention_config = aphrodite_config.attention_config if aphrodite_config is not None else None
    if attention_config is not None and attention_config.mla_prefill_backend is not None:
        selected_backend = attention_config.mla_prefill_backend
        backend_cls: type[MLAPrefillBackend] | None = None
        try:
            backend_cls = selected_backend.get_class()
            device_capability = current_platform.get_device_capability()
            invalid_reasons = backend_cls.validate_configuration(device_capability, selector_config)
        except ImportError:
            invalid_reasons = ["ImportError"]
        if invalid_reasons:
            raise ValueError(
                f"Selected MLA prefill backend {selected_backend.name} "
                f"is not valid for this configuration. "
                f"Reason: {invalid_reasons}"
            )
        assert backend_cls is not None
        logger.info_once("Using %s MLA prefill backend.", selected_backend.name)
        return backend_cls

    return _auto_select_mla_prefill_backend(selector_config)


@cache
def _auto_select_mla_prefill_backend(
    selector_config: MLAPrefillSelectorConfig,
) -> "type[MLAPrefillBackend]":
    """Auto-select the best available MLA prefill backend.

    Args:
        selector_config: Hashable configuration for backend selection.

    Returns:
        The selected prefill backend class.

    Raises:
        ValueError: If the platform has no valid MLA prefill backend.

    """
    from aphrodite.platforms import current_platform

    selected_backend = current_platform.get_mla_prefill_backend_cls(selector_config)
    logger.info_once("Using %s MLA prefill backend.", selected_backend.get_name())
    return selected_backend
