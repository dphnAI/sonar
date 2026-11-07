# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic multimodal helpers for Sonar Metal."""

from __future__ import annotations

from sonar.metal.multimodal.embeddings import merge_multimodal_embeddings
from sonar.metal.multimodal.feature_spec import (
    MultiModalFeatureSpec,
    PlaceholderRange,
)

__all__ = [
    "MultiModalFeatureSpec",
    "PlaceholderRange",
    "merge_multimodal_embeddings",
]
