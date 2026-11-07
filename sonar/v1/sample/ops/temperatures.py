# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

_SAMPLING_EPS = 1e-5


def apply_all_temperatures(
    logits: torch.Tensor,
    sampling_metadata,
) -> torch.Tensor:
    temp = sampling_metadata.temperature
    if sampling_metadata.dynatemp_min is not None and sampling_metadata.dynatemp_max is not None:
        mins = sampling_metadata.dynatemp_min
        maxs = sampling_metadata.dynatemp_max
        exps = sampling_metadata.dynatemp_exp
        assert exps is not None
        logprobs = logits.log_softmax(dim=-1)
        entropies = -(logprobs.exp() * logprobs).nansum(dim=-1)
        max_entropies = (logits > float("-inf")).sum(dim=-1).float().log_()
        normalized = torch.where(max_entropies > 0, entropies / max_entropies, 0.0)
        dynamic = mins + (maxs - mins) * normalized.pow(exps)
        temp = torch.where((mins != 0) | (maxs != 0), dynamic, temp)

    # Do not mutate request temperatures: greedy/random selection happens later,
    # and temperature storage is reused across steps.
    temp = torch.nan_to_num(temp, nan=_SAMPLING_EPS).clamp_min(_SAMPLING_EPS)
    if not sampling_metadata.all_random:
        temp = torch.where(sampling_metadata.temperature < _SAMPLING_EPS, 1.0, temp)
    return logits.div_(temp.unsqueeze(-1))
