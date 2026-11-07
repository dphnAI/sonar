# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def quadratic(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    Applies a quadratic transformation to the logits based on the
    provided smoothing factors and curves. The transformation is
    centered around the maximum logit value in the batch.
    The transformation involves a quadratic and cubic term, with the
    cubic term controlled by the smoothing curve. The quadratic term is
    scaled by the smoothing factor, and the cubic term is scaled by the
    product of the smoothing factor and the smoothing curve.
    params:
        logits (torch.Tensor): The logits to be transformed.
        smoothing_factors (torch.Tensor): The factors to scale the quadratic
            term in the transformation.
        smoothing_curves (torch.Tensor): The factors to scale the cubic term
            in the transformation.
    returns:
        torch.Tensor: The transformed logits.
    Credits: @kalomaze
    """
    smoothing_factor = sampling_metadata.quadratic_smoothing_factor
    smoothing_curve = sampling_metadata.quadratic_smoothing_curve
    if smoothing_factor is None or smoothing_curve is None:
        return logits

    mask = smoothing_factor != 0

    k = (smoothing_factor * (3 - smoothing_curve) / 2).unsqueeze_(dim=1)
    s = (smoothing_factor * (smoothing_curve - 1) / 2).unsqueeze_(dim=1)

    quadlogits = logits[mask]  # limit to logits using this sampler
    max_logits = quadlogits.max(dim=-1, keepdim=True).values

    # Construct the delta from each logit to its new value
    diff = quadlogits - max_logits
    diff -= diff**2 * (s[mask] * diff - k[mask])
    diff[diff != diff] = 0  # Eliminate NaNs due to infs

    logits[mask] -= diff
    return logits
