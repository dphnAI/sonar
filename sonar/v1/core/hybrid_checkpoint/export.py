# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-chunk Mamba state export (`mamba_prefill_export`).

In align mode a non-final prefill chunk ends at every position where a
Mamba state must be cached (a block boundary after a mid-block start, the
last cacheable boundary, a shared-prefix junction), so an early stop costs a
forward pass with a short chunk. With export, a non-final chunk may run past
one such position and have the backend write the state there with its
prefill checkpoint, which the scheduler, `MambaManager` and the worker all
take from the scheduler's per-request position. A chunk that the aligned
split cannot fill at all may instead end on a 64-token boundary.
"""

from typing import TYPE_CHECKING

from sonar.v1.kv_cache_interface import is_mamba_prefill_checkpoint_valid

if TYPE_CHECKING:
    from sonar.config import SonarConfig

# A filler chunk ends on a multiple of this many tokens and has at least
# _FILLER_MIN_TOKENS of them.
_FILLER_ALIGNMENT = 64
_FILLER_MIN_TOKENS = 1024


def plan_export(
    start: int,
    candidate_end: int,
    stops: tuple[int, ...],
    *,
    alignment: int | None,
    hash_block_size: int,
    block_size: int,
) -> tuple[int, int | None]:
    """End a non-final chunk so that it contains at most one of `stops`.

    `candidate_end` is where the chunk would end without stops. The first
    stop inside it is exported when the backend's checkpoint can write it
    (`is_mamba_prefill_checkpoint_valid`); the chunk then ends at the
    second stop, if any. A first stop that cannot be written ends the chunk.

    Returns:
        The chunk end and the exported position, or None.

    """
    if candidate_end <= start:
        return start, None
    interior = sorted(pos for pos in stops if start < pos < candidate_end)
    if not interior:
        return candidate_end, None
    first = interior[0]
    end = candidate_end if len(interior) == 1 else interior[1]
    if alignment is None or (first - start) % alignment:
        return first, None
    if not is_mamba_prefill_checkpoint_valid(
        query_start=start,
        query_end=end,
        checkpoint_position=first,
        hash_block_size=hash_block_size,
        mamba_block_size=block_size,
        checkpoint_alignment=alignment,
    ):
        return first, None
    return end, first


def plan_filler(
    start: int,
    num_new_tokens: int,
    *,
    prefill_end: int,
    stops: tuple[int, ...],
) -> int:
    """Tokens of a filler chunk for a non-final chunk the aligned split could
    not schedule: up to `num_new_tokens`, ending on a 64-token boundary
    before any of `stops`, or 0 when that leaves fewer than 1024."""
    if start >= prefill_end or num_new_tokens >= prefill_end - start:
        return 0
    take = num_new_tokens // _FILLER_ALIGNMENT * _FILLER_ALIGNMENT
    inner = [pos for pos in stops if start < pos < start + take]
    if take > 0 and inner:
        take = (min(inner) - start) // _FILLER_ALIGNMENT * _FILLER_ALIGNMENT
    return take if take >= _FILLER_MIN_TOKENS else 0


def check_prefill_export_config(
    sonar_config: "SonarConfig",
    *,
    use_v2_model_runner: bool,
    checkpoint_alignment: int | None,
    has_checkpoint_blocks: bool,
) -> None:
    """Raise unless in-chunk export can run with this configuration."""
    refusal = None
    if sonar_config.cache_config.mamba_cache_mode != "align":
        refusal = "prefix caching with mamba_cache_mode 'align'"
    elif not use_v2_model_runner:
        refusal = "Model Runner V2"
    elif checkpoint_alignment is None or not has_checkpoint_blocks:
        refusal = "a backend that writes prefill checkpoints in every Mamba group"
    elif sonar_config.speculative_config is not None and sonar_config.speculative_config.use_eagle_block_drop():
        # An exported position follows neither the EAGLE block drop nor the
        # lookahead block hashes that replace it.
        refusal = "speculative_config.disable_eagle_block_drop"
    elif sonar_config.parallel_config.use_ubatching:
        refusal = "no ubatching"
    elif (
        sonar_config.parallel_config.decode_context_parallel_size != 1
        or sonar_config.parallel_config.prefill_context_parallel_size != 1
    ):
        refusal = "decode and prefill context parallel size 1"
    if refusal is not None:
        raise ValueError(f"mamba_prefill_export needs {refusal}")
