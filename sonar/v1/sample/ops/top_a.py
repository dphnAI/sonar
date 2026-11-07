# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def top_a(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    top_a = sampling_metadata.top_a
    if top_a is None:
        return logits

    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    threshold = torch.pow(top_probs, 2) * top_a.unsqueeze(dim=1)
    tokens_to_remove = probs < threshold
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits
