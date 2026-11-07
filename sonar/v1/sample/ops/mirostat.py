# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mirostat sampling operation for v1 engine."""

import torch
from torch import Tensor

from aphrodite.v1.sample.metadata import SamplingMetadata


def _fetch_mirostat_args(
    sampling_metadata: SamplingMetadata,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Extract mirostat parameters for applicable requests.

    Returns:
        Tuple of (logit_indices, taus, etas, mus)
    """
    logit_indices: list[int] = []
    taus: list[float] = []
    etas: list[float] = []
    mus: list[float] = []

    if (
        sampling_metadata.mirostat_mode is None
        or sampling_metadata.mirostat_tau is None
        or sampling_metadata.mirostat_eta is None
    ):
        return logit_indices, taus, etas, mus

    batch_size = len(sampling_metadata.output_token_ids)

    for req_idx in range(batch_size):
        mode = sampling_metadata.mirostat_mode[req_idx].item()
        if mode == 2:  # Mirostat v2
            logit_indices.append(req_idx)
            taus.append(sampling_metadata.mirostat_tau[req_idx].item())
            etas.append(sampling_metadata.mirostat_eta[req_idx].item())
            # Get current mu from persistent data, or initialize
            mu = sampling_metadata.persistent_data.get(req_idx, {}).get(
                "miro_mu", sampling_metadata.mirostat_tau[req_idx].item() * 2
            )
            mus.append(mu)

    return logit_indices, taus, etas, mus


def _store_mirostat_args(req_indices: list[int], mus: list[float], sampling_metadata: SamplingMetadata) -> None:
    """Store updated mu values back to persistent data."""
    for req_idx, mu in zip(req_indices, mus):
        if req_idx not in sampling_metadata.persistent_data:
            sampling_metadata.persistent_data[req_idx] = {}
        sampling_metadata.persistent_data[req_idx]["miro_mu"] = mu


def _apply_mirostat_v2(
    logits: Tensor,
    taus: list[float],  # Target surprisal
    etas: list[float],  # Learning rate
    mus: list[float],  # Current mu accumulator
) -> Tensor:
    """Apply Mirostat v2 algorithm to logits.

    Args:
        logits: Input logits tensor [batch_size, vocab_size]
        taus: Target surprisal values for each request
        etas: Learning rate for each request
        mus: Current mu values for each request (in-out parameter)

    Returns:
        Modified logits tensor
    """
    if not taus:  # No mirostat requests
        return logits

    ttaus = torch.tensor(taus, dtype=logits.dtype, device=logits.device)
    tetas = torch.tensor(etas, dtype=logits.dtype, device=logits.device)
    tmus = torch.tensor(mus, dtype=logits.dtype, device=logits.device)

    logit_surprise = torch.softmax(logits, dim=-1).log2_().neg_()

    # too surprising - mask out tokens
    miro_mask = logit_surprise > tmus.unsqueeze(dim=-1)

    # ensure at least one token is selectable (most likely one)
    mininds = torch.argmin(logit_surprise, dim=-1)
    miro_mask.scatter_(1, mininds.unsqueeze(dim=-1), False)

    logits[miro_mask] = -float("inf")

    probs = torch.softmax(logits, dim=-1, dtype=logits.dtype)
    next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True)

    picked_surprises = torch.gather(logit_surprise, dim=-1, index=next_token_ids)
    eps = picked_surprises.squeeze() - ttaus  # error from target
    tmus = tmus - tetas * eps  # update mu with learning rate

    mus[:] = tmus.tolist()

    # one-hot conversion for deterministic output
    logits.fill_(-float("inf"))
    logits.scatter_(1, next_token_ids, 1.0)

    return logits


def mirostat(
    logits: Tensor,
    sampling_metadata: SamplingMetadata,
) -> Tensor:
    """Apply mirostat sampling if any requests have it enabled.

    Args:
        logits: Input logits tensor
        sampling_metadata: Sampling metadata containing mirostat params

    Returns:
        Modified logits tensor
    """
    if (
        sampling_metadata.mirostat_mode is None
        or sampling_metadata.mirostat_tau is None
        or sampling_metadata.mirostat_eta is None
    ):
        return logits

    batch_size = len(sampling_metadata.output_token_ids)

    has_mirostat = any(sampling_metadata.mirostat_mode[i].item() == 2 for i in range(batch_size))

    if not has_mirostat:
        return logits

    logit_indices, taus, etas, mus = _fetch_mirostat_args(sampling_metadata)

    if not logit_indices:  # No mirostat requests
        return logits

    logits[logit_indices] = _apply_mirostat_v2(logits[logit_indices], taus, etas, mus)

    _store_mirostat_args(logit_indices, mus, sampling_metadata)

    return logits
