# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def epsilon_cutoff(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    epsilon_cutoff = sampling_metadata.epsilon_cutoff
    if epsilon_cutoff is None:
        return logits

    probs = logits.softmax(dim=-1)

    eps_mask = probs < epsilon_cutoff.unsqueeze(dim=1)

    # guard against nulling out all the logits
    top_idx = torch.argmax(probs, dim=1, keepdim=True)
    eps_mask.scatter_(dim=1, index=top_idx, value=False)

    logits[eps_mask] = -float("inf")
    return logits
