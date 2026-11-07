# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def eta_cutoff(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    eta_cutoff = sampling_metadata.eta_cutoff
    if eta_cutoff is None:
        return logits

    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1)
    eps = torch.min(eta_cutoff, torch.sqrt(eta_cutoff) * torch.exp(neg_entropy)).unsqueeze(dim=1)

    eta_mask = probs < eps

    # guard against nulling out all the logits
    top_idx = torch.argmax(probs, dim=1, keepdim=True)
    eta_mask.scatter_(dim=1, index=top_idx, value=False)

    logits[eta_mask] = -float("inf")
    return logits
