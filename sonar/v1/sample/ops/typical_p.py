# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def typical_p(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    typical_p = sampling_metadata.typical_p
    if typical_p is None:
        return logits

    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1, keepdim=True)

    surprisal_deviations = (neg_entropy - shifted_logits).abs()
    _, indices = torch.sort(surprisal_deviations)
    reordered_probs = probs.gather(-1, indices)
    typ_mask_sorted = reordered_probs.cumsum(dim=-1) >= typical_p.unsqueeze(dim=1)

    min_tokens_to_keep = 1
    # Keep at least min_tokens_to_keep
    typ_mask_sorted[..., :min_tokens_to_keep] = 0

    typ_mask = typ_mask_sorted.scatter(1, indices, typ_mask_sorted)
    logits[typ_mask] = -float("inf")
    return logits
