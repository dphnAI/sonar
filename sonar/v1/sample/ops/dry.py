# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any

import torch

DRY_STATE_KEY = "dry_state"


@dataclass
class DryRequestState:
    history: list[int]
    breaker_ids: frozenset[int]
    positions_by_token: dict[int, list[int]] = field(default_factory=dict)

    @classmethod
    def from_history(cls, tokens: list[int], breaker_ids: list[int]) -> "DryRequestState":
        positions_by_token: dict[int, list[int]] = {}
        for idx, token in enumerate(tokens):
            positions_by_token.setdefault(token, []).append(idx)
        return cls(
            history=list(tokens),
            breaker_ids=frozenset(breaker_ids),
            positions_by_token=positions_by_token,
        )

    def append_token(self, token: int) -> None:
        if token < 0:
            return
        self.positions_by_token.setdefault(token, []).append(len(self.history))
        self.history.append(token)

    def extend_tokens(self, tokens: list[int]) -> None:
        for token in tokens:
            self.append_token(token)


def init_dry_state(
    persistent_data: dict[str, Any],
    prompt_token_ids: list[int] | None,
    output_token_ids: list[int],
    breaker_ids: list[int],
) -> None:
    persistent_data[DRY_STATE_KEY] = DryRequestState.from_history(
        (prompt_token_ids or []) + output_token_ids,
        breaker_ids,
    )


def update_dry_state(
    persistent_data: dict[str, Any],
    token_ids: list[int],
) -> None:
    dry_state = persistent_data.get(DRY_STATE_KEY)
    if isinstance(dry_state, DryRequestState):
        dry_state.extend_tokens(token_ids)


def _get_or_rebuild_dry_state(
    persistent_data: dict[str, Any],
    prompt_token_ids: list[int] | None,
    output_token_ids: list[int],
    breaker_ids: list[int],
) -> DryRequestState:
    history = (prompt_token_ids or []) + output_token_ids
    dry_state = persistent_data.get(DRY_STATE_KEY)
    if (
        not isinstance(dry_state, DryRequestState)
        or dry_state.breaker_ids != frozenset(breaker_ids)
        or len(dry_state.history) != len(history)
        or (history and dry_state.history[-1] != history[-1])
    ):
        dry_state = DryRequestState.from_history(history, breaker_ids)
        persistent_data[DRY_STATE_KEY] = dry_state
    return dry_state


def _compute_dry_penalties(
    dry_state: DryRequestState,
    allowed_length: int,
    range_limit: int,
    max_ngram: int,
    max_occurrences: int,
    early_exit_match_len: int,
) -> dict[int, int]:
    history = dry_state.history
    history_len = len(history)
    if history_len < 2:
        return {}

    last_token = history[-1]
    if last_token in dry_state.breaker_ids:
        return {}

    start_idx = max(0, history_len - range_limit) if range_limit > 0 else 0
    endpoint_positions = dry_state.positions_by_token.get(last_token, [])
    start_offset = bisect_left(endpoint_positions, start_idx)
    endpoint_indexes = endpoint_positions[start_offset:-1]
    if not endpoint_indexes:
        return {}

    if len(endpoint_indexes) > max_occurrences:
        endpoint_indexes = endpoint_indexes[-max_occurrences:]

    curr_max_ngram = 0
    for curr_max_ngram in range(min(history_len - start_idx, max_ngram + 1)):
        if history[history_len - curr_max_ngram - 1] in dry_state.breaker_ids:
            break

    if curr_max_ngram <= allowed_length:
        return {}

    penalties: dict[int, int] = {}
    for idx in reversed(endpoint_indexes):
        match_len = 0
        max_unwind = min(idx - start_idx, curr_max_ngram)
        for unwind in range(1, max_unwind + 1):
            candidate_tok = history[idx - unwind]
            if candidate_tok in dry_state.breaker_ids:
                break
            if candidate_tok != history[history_len - unwind - 1]:
                break
            match_len = unwind

        if match_len <= 0:
            continue

        next_tok = history[idx + 1]
        new_len = match_len + 1
        penalties[next_tok] = max(penalties.get(next_tok, 0), new_len)
        if new_len >= early_exit_match_len:
            break

    return penalties


def _apply_dry_to_row(
    logits: torch.Tensor,
    token_seq: torch.Tensor,
    multiplier: torch.Tensor,
    base: torch.Tensor,
    min_ngram: int,
    row_breaker_ids: torch.Tensor,
    curr_max_ngram: int,
    max_occurrences_val: int,
    early_exit_match_len_val: int,
    vocab_size: int,
    row_index: int,
) -> None:
    if token_seq.size(0) < 2:
        return

    valid_breaker_ids = row_breaker_ids[row_breaker_ids != vocab_size]
    last_token = token_seq[-1]
    if valid_breaker_ids.numel() != 0:
        if (valid_breaker_ids == last_token).any():
            return
        break_mask = (token_seq.unsqueeze(1) == valid_breaker_ids.unsqueeze(0)).any(dim=1)
    else:
        break_mask = torch.zeros(len(token_seq), dtype=torch.bool, device=logits.device)

    for curr_max_ngram in range(min(len(break_mask), curr_max_ngram + 1)):  # noqa: B020
        if break_mask[-curr_max_ngram - 1]:
            break

    if curr_max_ngram <= min_ngram:
        return

    endpoint_indexes = torch.nonzero(token_seq == last_token, as_tuple=True)[0]
    if endpoint_indexes.numel() < 2:
        return
    endpoint_indexes = endpoint_indexes[:-1]

    if endpoint_indexes.numel() > max_occurrences_val:
        endpoint_indexes = endpoint_indexes[-max_occurrences_val:]

    reversed_endpoint_indexes = endpoint_indexes.flip(0)
    match_cap = torch.minimum(
        reversed_endpoint_indexes,
        torch.full_like(reversed_endpoint_indexes, curr_max_ngram),
    )
    max_match_cap = match_cap.max().item()
    if max_match_cap == 0:
        return

    offsets = torch.arange(1, max_match_cap + 1, device=logits.device, dtype=torch.long)
    candidate_positions = reversed_endpoint_indexes.unsqueeze(1) - offsets
    suffix_positions = token_seq.size(0) - 1 - offsets
    valid_offsets = offsets.unsqueeze(0) <= match_cap.unsqueeze(1)
    matching_steps = (
        valid_offsets
        & ~break_mask[candidate_positions]
        & (token_seq[candidate_positions] == token_seq[suffix_positions])
    )
    match_lens = matching_steps.to(torch.int64).cumprod(dim=1).sum(dim=1)
    matched_mask = match_lens > 0
    if not matched_mask.any():
        return

    matched_indexes = reversed_endpoint_indexes[matched_mask]
    new_lens = match_lens[matched_mask] + 1

    early_exit_hits = torch.nonzero(new_lens >= early_exit_match_len_val, as_tuple=True)[0]
    if early_exit_hits.numel():
        cutoff = early_exit_hits[0].item() + 1
        matched_indexes = matched_indexes[:cutoff]
        new_lens = new_lens[:cutoff]

    penalized_tokens_t = token_seq[matched_indexes + 1]
    unique_tokens, inverse = penalized_tokens_t.unique(return_inverse=True)
    max_lens = torch.zeros(unique_tokens.size(0), dtype=torch.int64, device=logits.device)
    max_lens.scatter_reduce_(0, inverse, new_lens, reduce="amax", include_self=False)
    scales = base ** (max_lens - min_ngram)
    logits[row_index, unique_tokens] -= multiplier * scales


def apply_all_dry(
    logits: torch.Tensor,
    input_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    bases: torch.Tensor,
    allowed_lengths: torch.Tensor,
    sequence_breakers_ids: torch.Tensor,
    ranges: torch.Tensor,
    max_ngram: torch.Tensor,
    max_occurrences: torch.Tensor,
    early_exit_match_len: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Don't Repeat Yourself (DRY) sampling to the logits.

    Reference: https://github.com/oobabooga/text-generation-webui/pull/5677 and
    https://github.com/AlpinDale/aphrodite/pull/1
    """
    vocab_size = logits.size(-1)

    applies_to = multipliers.nonzero(as_tuple=True)[0]
    for irow_t in applies_to:
        irow = irow_t.item()
        prompt_row = input_token_ids[irow]
        output_row = output_token_ids[irow]
        prompt_len = (prompt_row != vocab_size).sum().item()
        output_len = (output_row != vocab_size).sum().item()

        token_seq = torch.cat((prompt_row[:prompt_len], output_row[:output_len]), dim=0)

        range_limit = ranges[irow].item()
        if range_limit > 0:
            token_seq = token_seq[-range_limit:]

        _apply_dry_to_row(
            logits=logits,
            token_seq=token_seq,
            multiplier=multipliers[irow],
            base=bases[irow],
            min_ngram=allowed_lengths[irow].item(),
            row_breaker_ids=sequence_breakers_ids[irow],
            curr_max_ngram=max_ngram[irow].item(),
            max_occurrences_val=max_occurrences[irow].item(),
            early_exit_match_len_val=early_exit_match_len[irow].item(),
            vocab_size=vocab_size,
            row_index=irow,
        )

    return logits


def apply_all_dry_history(
    logits: torch.Tensor,
    token_history_ids: torch.Tensor,
    token_history_lens: torch.Tensor,
    multipliers: torch.Tensor,
    bases: torch.Tensor,
    allowed_lengths: torch.Tensor,
    sequence_breakers_ids: torch.Tensor,
    ranges: torch.Tensor,
    max_ngram: torch.Tensor,
    max_occurrences: torch.Tensor,
    early_exit_match_len: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.size(-1)

    applies_to = multipliers.nonzero(as_tuple=True)[0]
    for irow_t in applies_to:
        irow = irow_t.item()
        token_seq = token_history_ids[irow, : token_history_lens[irow].item()]
        range_limit = ranges[irow].item()
        if range_limit > 0:
            token_seq = token_seq[-range_limit:]

        _apply_dry_to_row(
            logits=logits,
            token_seq=token_seq,
            multiplier=multipliers[irow],
            base=bases[irow],
            min_ngram=allowed_lengths[irow].item(),
            row_breaker_ids=sequence_breakers_ids[irow],
            curr_max_ngram=max_ngram[irow].item(),
            max_occurrences_val=max_occurrences[irow].item(),
            early_exit_match_len_val=early_exit_match_len[irow].item(),
            vocab_size=vocab_size,
            row_index=irow,
        )

    return logits
