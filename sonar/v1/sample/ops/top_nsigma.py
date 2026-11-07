# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def top_nsigma(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Apply top-nsigma truncation to the logits.

    Reference: https://arxiv.org/abs/2411.07641

    Args:
        logits: Logits of shape (num_tokens, vocab_size)
        nsigma: Number of standard deviations to use as threshold
    Returns:
        Modified logits with values below threshold set to -inf
    """
    nsigma = sampling_metadata.top_nsigma
    if nsigma is None:
        return logits

    # Grammar, phrase-retry and no-repeat masks are not part of the support
    # whose dispersion defines the cutoff.
    finite = logits.isfinite()
    count = finite.sum(dim=-1, keepdim=True)
    values = logits.masked_fill(~finite, 0)
    mean = values.sum(dim=-1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(finite, logits - mean, 0)
    std = (centered.square().sum(dim=-1, keepdim=True) / (count - 1).clamp_min(1)).sqrt_()
    maximum = logits.masked_fill(~finite, float("-inf")).amax(dim=-1, keepdim=True)
    threshold = maximum - nsigma.unsqueeze(-1) * std
    logits.masked_fill_((nsigma > 0).unsqueeze(-1) & (logits < threshold), float("-inf"))

    return logits
