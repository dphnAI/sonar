# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod

import torch

from aphrodite.config import AphroditeConfig
from aphrodite.v1.attention.backend import AttentionBackend, AttentionImpl
from aphrodite.v1.kv_cache_interface import KVCacheSpec


class AttentionLayerBase(ABC):
    """Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"
    # Set by the speculator on its draft model's layers.
    is_draft_layer: bool = False

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        self.kv_cache = kv_cache

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        pass

    @abstractmethod
    def get_kv_cache_spec(self, aphrodite_config: AphroditeConfig) -> KVCacheSpec | None:
        """Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
