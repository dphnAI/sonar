# SPDX-License-Identifier: Apache-2.0
"""Generic multimodal helpers for Aphrodite Metal."""

from __future__ import annotations

from aphrodite.metal.multimodal.embeddings import merge_multimodal_embeddings
from aphrodite.metal.multimodal.feature_spec import (
    MultiModalFeatureSpec,
    PlaceholderRange,
)

__all__ = [
    "MultiModalFeatureSpec",
    "PlaceholderRange",
    "merge_multimodal_embeddings",
]
