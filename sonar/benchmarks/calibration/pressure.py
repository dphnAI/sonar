# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose how many concurrent sessions put the KV cache under pressure.

Eviction order only matters when idle sessions compete for cache space.
With too few sessions everything fits; with too many, running requests
evict each other's prefixes before any are reused, and no order helps. The
probe runs the workload at a few session counts around the cache capacity
and keeps the largest count whose prefix hit rate is still a large share of
the best one seen."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

MULTIPLIERS = (0.5, 1.0, 1.5, 2.0)
MIN_SESSIONS, MAX_SESSIONS = 2, 512
# Keep a count whose hit rate is at least this share of the best probe.
KEEP_SHARE = 0.6
# A probe below this share of the best has collapsed; larger counts are skipped.
COLLAPSE_SHARE = 0.3
# Below this best hit rate the workload barely reuses the cache at all.
MIN_USEFUL_HIT_RATE = 0.05


@dataclass(frozen=True, slots=True)
class PressureStep:
    sessions: int
    prefix_hit_rate: float | None
    output_tokens_per_second: float
    ttft_p90: float


def ladder(kv_capacity_tokens: int, mean_session_context: float) -> list[int]:
    """Session counts whose contexts total half to twice the cache."""
    if kv_capacity_tokens <= 0 or mean_session_context <= 0:
        raise ValueError("The KV cache capacity and mean session context must be positive.")
    base = kv_capacity_tokens / mean_session_context
    counts = {min(max(math.ceil(base * multiplier), MIN_SESSIONS), MAX_SESSIONS) for multiplier in MULTIPLIERS}
    return sorted(counts)


def _rate(step: PressureStep) -> float:
    return step.prefix_hit_rate or 0.0


def next_sessions(counts: Sequence[int], steps: Sequence[PressureStep]) -> int | None:
    """The next count to probe, in ascending order, or None when done."""
    if steps:
        best = max(_rate(step) for step in steps)
        if best >= MIN_USEFUL_HIT_RATE and _rate(steps[-1]) < COLLAPSE_SHARE * best:
            return None
    done = {step.sessions for step in steps}
    return next((count for count in counts if count not in done), None)


def choose_sessions(steps: Sequence[PressureStep]) -> tuple[int, str]:
    """The count to calibrate with, and a sentence explaining it."""
    if not steps:
        raise ValueError("No pressure probes were run.")
    best = max(steps, key=_rate)
    if _rate(best) < MIN_USEFUL_HIT_RATE:
        smallest = min(steps, key=lambda step: step.sessions)
        return smallest.sessions, (
            f"Even {smallest.sessions} sessions reused only {_rate(smallest) * 100:.1f}% of prompt tokens from the "
            "cache, so eviction has little to decide. Check the KV cache size and --max-model-len."
        )
    kept = [step for step in steps if _rate(step) >= KEEP_SHARE * _rate(best)]
    chosen = max(kept, key=lambda step: step.sessions)
    if chosen is best:
        reason = f"{chosen.sessions} sessions had the most prefix hits ({_rate(chosen) * 100:.0f}%)."
    else:
        reason = (
            f"{chosen.sessions} sessions kept {_rate(chosen) * 100:.0f}% prefix hits under pressure "
            f"({best.sessions} sessions: {_rate(best) * 100:.0f}%)."
        )
    return chosen.sessions, reason
