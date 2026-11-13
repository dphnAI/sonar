# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit the FlashStates prefill cost model to measured prefill times.

The model is `T(L) = C + D * L + E * L**2 / 200000`. `D` and `E` are the
`--prefill-cost-coefficients` that GreedyDual-Size eviction uses, and `C`
absorbs fixed per-request overhead that Sonar does not need.
"""

import itertools
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_QUADRATIC_SCALE = 2e5
# A per-token cost this small means the timings carry no length signal.
_MIN_PER_TOKEN = 1e-15


@dataclass(frozen=True, slots=True)
class PrefillSample:
    tokens: int
    seconds: float


@dataclass(frozen=True, slots=True)
class CostFit:
    per_token: float
    per_token_per_100k: float
    intercept: float
    ratio: float
    ratio_interval: tuple[float, float]
    r_squared: float
    max_relative_error: float
    num_samples: int
    min_tokens: int
    max_tokens: int

    def predict(self, tokens: int | float) -> float:
        return self.intercept + self.per_token * tokens + self.per_token_per_100k * tokens * tokens / _QUADRATIC_SCALE

    def coefficients(self, ratio: float) -> tuple[float, float]:
        """`(D, ratio * D)`, as passed to `--prefill-cost-coefficients`."""
        return self.per_token, ratio * self.per_token


class FitError(ValueError):
    """The measurements cannot support a fit. The message says what to change."""


def _design(tokens: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(tokens), tokens, tokens * tokens / _QUADRATIC_SCALE], axis=1)


def _solve(tokens: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    """Minimise the relative squared error with every coefficient >= 0.

    With three coefficients the constrained optimum is the unconstrained
    optimum of one support set, so trying every support is exact."""
    design = _design(tokens) / seconds[:, None]
    target = np.ones_like(seconds)
    best: tuple[float, np.ndarray] | None = None
    for size in (3, 2, 1):
        for support in itertools.combinations(range(3), size):
            if 1 not in support:
                continue
            columns = design[:, support]
            scale = np.linalg.norm(columns, axis=0)
            if np.any(scale == 0):
                continue
            solution, *_ = np.linalg.lstsq(columns / scale, target, rcond=None)
            solution = solution / scale
            if np.any(solution < 0):
                continue
            coefficients = np.zeros(3)
            coefficients[list(support)] = solution
            residual = float(np.sum((design @ coefficients - target) ** 2))
            if best is None or residual < best[0] - 1e-15:
                best = (residual, coefficients)
    if best is None or best[1][1] <= _MIN_PER_TOKEN:
        raise FitError(
            "Prefill time does not increase with prompt length in these measurements. "
            "Profile longer prompts (raise --profile-max-tokens) or check that requests were not served from cache."
        )
    return best[1]


def _validate(samples: Sequence[PrefillSample]) -> None:
    lengths = {sample.tokens for sample in samples}
    if len(lengths) < 3:
        raise FitError(f"Need prefill timings at 3 or more distinct prompt lengths, got {len(lengths)}.")
    for sample in samples:
        if sample.tokens <= 0 or not math.isfinite(sample.seconds) or sample.seconds <= 0:
            raise FitError(f"Invalid prefill timing: {sample.seconds!r} seconds for {sample.tokens} tokens.")


def _medians(groups: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = sorted(groups)
    return (
        np.array(lengths, dtype=np.float64),
        np.array([np.median(groups[length]) for length in lengths], dtype=np.float64),
    )


def fit_prefill_cost(samples: Sequence[PrefillSample], *, bootstrap: int = 500, seed: int = 0) -> CostFit:
    """Fit `C`, `D` and `E` with `C >= 0`, `D > 0` and `E >= 0` to the
    median time at each prompt length, so one slow request (a kernel
    compiled on first use, a stray stall) cannot bend the curve.

    The ratio interval is the 5th to 95th percentile of `E / D` over
    bootstrap resamples drawn within each prompt length."""
    _validate(samples)
    groups: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        groups[sample.tokens].append(sample.seconds)
    tokens, seconds = _medians(groups)
    intercept, per_token, quadratic = _solve(tokens, seconds)

    predicted = _design(tokens) @ np.array([intercept, per_token, quadratic])
    weights = 1.0 / seconds**2
    weighted_mean = float(np.sum(weights * seconds) / np.sum(weights))
    total = float(np.sum(weights * (seconds - weighted_mean) ** 2))
    residual = float(np.sum(weights * (predicted - seconds) ** 2))

    return CostFit(
        per_token=float(per_token),
        per_token_per_100k=float(quadratic),
        intercept=float(intercept),
        ratio=float(quadratic / per_token),
        ratio_interval=_bootstrap_ratio(groups, bootstrap, seed),
        r_squared=1.0 - residual / total if total > 0 else 1.0,
        max_relative_error=float(np.max(np.abs(predicted - seconds) / seconds)),
        num_samples=len(samples),
        min_tokens=int(tokens.min()),
        max_tokens=int(tokens.max()),
    )


def _bootstrap_ratio(groups: dict[int, list[float]], rounds: int, seed: int) -> tuple[float, float]:
    if rounds <= 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(rounds):
        resampled = {
            length: [values[i] for i in rng.integers(0, len(values), size=len(values))]
            for length, values in groups.items()
        }
        try:
            _, per_token, quadratic = _solve(*_medians(resampled))
        except FitError:
            continue
        ratios.append(quadratic / per_token)
    if len(ratios) < rounds / 2:
        return (0.0, math.inf)
    low, high = np.percentile(ratios, [5, 95])
    return (float(low), float(high))
