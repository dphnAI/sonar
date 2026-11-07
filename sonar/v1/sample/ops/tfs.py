# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.v1.sample.metadata import SamplingMetadata


def tfs(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    tfs = sampling_metadata.tfs
    if tfs is None:
        return logits

    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
    d2 = logits_sort.softmax(dim=-1).diff().diff().abs()
    normalized_d2 = d2 / torch.sum(d2, dim=-1, keepdim=True)
    curvature_cdf = torch.cumsum(normalized_d2, dim=-1)

    tfs_mask = curvature_cdf > tfs.unsqueeze(dim=-1)

    tfs_mask = torch.cat(
        (
            torch.zeros(logits.shape[0], 1, dtype=torch.bool, device=logits.device),
            tfs_mask,
            torch.ones(logits.shape[0], 1, dtype=torch.bool, device=logits.device),
        ),
        dim=-1,
    )

    logits_sort[tfs_mask] = -float("inf")
    logits = torch.gather(logits_sort, dim=-1, index=torch.argsort(logits_idx, dim=-1))

    return logits
