# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def min_p(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    Filters logits using adaptive probability thresholding.
    """
    min_p = getattr(sampling_metadata, "min_p", None)
    if min_p is None:
        return logits

    # Convert logits to probability distribution
    probability_values = torch.nn.functional.softmax(logits, dim=-1)
    # Calculate maximum probabilities per sequence
    max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
    # Reshape min_p for broadcasting
    adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
    # Identify valid tokens using threshold comparison
    valid_token_mask = probability_values >= adjusted_min_p
    # Apply mask using boolean indexing
    logits[~valid_token_mask] = -float("inf")
    return logits
