# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose eviction ratios to try and the ratio to recommend.

Only `R = E / D` changes GreedyDual-Size eviction. Every function here is a
pure function of the finished trials, so an interrupted calibration resumes
with the same decisions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Objective = Literal["throughput", "latency"]

_EXCLUSION_FACTOR = 1.25
_EXPANSION_FACTOR = 4.0
_DEFAULT_ANCHOR = 0.2
_MAX_FAILURE_FRACTION = 0.01
# Two-sided 97.5% Student t quantiles for 1..30 degrees of freedom.
_T_QUANTILES = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class TrialMeasurement:
    ratio: float
    output_tokens_per_second: float
    interval_output_tokens_per_second: tuple[float, ...]
    ttft_p50: float
    ttft_p90: float
    completed_requests: int
    failed_requests: int
    prefix_hit_rate: float | None
    frontier_hit_tokens: int | None
    kv_usage_p50: float | None


@dataclass(frozen=True, slots=True)
class SearchConfig:
    fitted_ratio: float
    fitted_interval: tuple[float, float]
    max_trials: int
    objective: Objective = "throughput"
    max_ttft_p90: float | None = None
    tolerance: float = 0.02
    min_ratio: float = 1e-3
    max_ratio: float = 100.0


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    ratio: float
    trials: int
    output_tokens_per_second: float
    ci95: tuple[float, float]
    ttft_p90: float
    score: float
    meets_slo: bool


@dataclass(frozen=True, slots=True)
class Selection:
    ratio: float
    candidates: tuple[CandidateSummary, ...]
    reason: str
    confident: bool
    note: str | None


def format_ratio(ratio: float) -> str:
    return "0" if ratio == 0 else f"{ratio:.3g}"


def _same_ratio(a: float, b: float) -> bool:
    if a == 0 or b == 0:
        return a == b
    return abs(a - b) <= 1e-9 * max(a, b)


def _t_quantile(degrees: int) -> float:
    return _T_QUANTILES[degrees - 1] if degrees <= len(_T_QUANTILES) else 1.96


@dataclass(frozen=True, slots=True)
class _Pooled:
    ratio: float
    trials: int
    throughput: float
    ci95: tuple[float, float]
    ttft_p90: float
    failed_fraction: float


def _pool(trials: Sequence[TrialMeasurement]) -> list[_Pooled]:
    groups: list[list[TrialMeasurement]] = []
    for trial in trials:
        for group in groups:
            if _same_ratio(group[0].ratio, trial.ratio):
                group.append(trial)
                break
        else:
            groups.append([trial])
    pooled = []
    for group in groups:
        samples = [rate for trial in group for rate in trial.interval_output_tokens_per_second]
        throughput = (
            sum(samples) / len(samples)
            if samples
            else sum(trial.output_tokens_per_second for trial in group) / len(group)
        )
        if len(samples) >= 2:
            variance = sum((rate - throughput) ** 2 for rate in samples) / (len(samples) - 1)
            half = _t_quantile(len(samples) - 1) * math.sqrt(variance / len(samples))
            ci95 = (throughput - half, throughput + half)
        else:
            ci95 = (-math.inf, math.inf)
        completed = sum(trial.completed_requests for trial in group)
        failed = sum(trial.failed_requests for trial in group)
        pooled.append(
            _Pooled(
                ratio=group[0].ratio,
                trials=len(group),
                throughput=throughput,
                ci95=ci95,
                ttft_p90=sum(trial.ttft_p90 for trial in group) / len(group),
                failed_fraction=failed / max(completed, 1),
            )
        )
    return pooled


def _summaries(config: SearchConfig, trials: Sequence[TrialMeasurement]) -> list[CandidateSummary]:
    pooled = _pool(trials)
    if not pooled:
        return []
    eligible = [entry for entry in pooled if entry.failed_fraction <= _MAX_FAILURE_FRACTION]
    best_throughput = max((entry.throughput for entry in eligible), default=0.0)
    summaries = []
    for entry in pooled:
        meets_slo = config.max_ttft_p90 is None or entry.ttft_p90 <= config.max_ttft_p90
        if entry.failed_fraction > _MAX_FAILURE_FRACTION:
            score = 0.0 if config.objective == "throughput" else -math.inf
        elif config.objective == "throughput":
            score = entry.throughput
            if not meets_slo:
                assert config.max_ttft_p90 is not None
                score *= config.max_ttft_p90 / entry.ttft_p90
        else:
            floor = (1 - config.tolerance) * best_throughput
            shortfall = min(1.0, entry.throughput / floor) if floor > 0 else 1.0
            score = -entry.ttft_p90 / max(shortfall, 1e-9)
        summaries.append(
            CandidateSummary(
                ratio=entry.ratio,
                trials=entry.trials,
                output_tokens_per_second=entry.throughput,
                ci95=entry.ci95,
                ttft_p90=entry.ttft_p90,
                score=score,
                meets_slo=meets_slo,
            )
        )
    return summaries


def _clamp(config: SearchConfig, ratio: float) -> float:
    return min(max(ratio, config.min_ratio), config.max_ratio)


def _admissible(ratio: float, summaries: Sequence[CandidateSummary]) -> bool:
    if ratio == 0:
        return not any(summary.ratio == 0 for summary in summaries)
    for summary in summaries:
        if summary.ratio > 0 and max(ratio, summary.ratio) / min(ratio, summary.ratio) < _EXCLUSION_FACTOR:
            return False
    return True


def _overlap(a: CandidateSummary, b: CandidateSummary) -> bool:
    return a.ci95[0] <= b.ci95[1] and b.ci95[0] <= a.ci95[1]


def _anchor(config: SearchConfig) -> float:
    return _clamp(config, config.fitted_ratio) if config.fitted_ratio > config.min_ratio else _DEFAULT_ANCHOR


def _refine(config: SearchConfig, summaries: list[CandidateSummary]) -> float | None:
    positive = sorted((summary for summary in summaries if summary.ratio > 0), key=lambda summary: summary.ratio)
    if not positive:
        return None
    index = max(range(len(positive)), key=lambda i: positive[i].score)
    best = positive[index]
    if index == 0 and best.ratio / _EXPANSION_FACTOR >= config.min_ratio * (1 - 1e-12):
        candidate = _clamp(config, best.ratio / _EXPANSION_FACTOR)
        if _admissible(candidate, summaries):
            return candidate
    if index == len(positive) - 1 and best.ratio * _EXPANSION_FACTOR <= config.max_ratio * (1 + 1e-12):
        candidate = _clamp(config, best.ratio * _EXPANSION_FACTOR)
        if _admissible(candidate, summaries):
            return candidate
    left = positive[index - 1] if index > 0 else None
    right = positive[index + 1] if index + 1 < len(positive) else None
    if left is not None and right is not None:
        xs = [math.log(entry.ratio) for entry in (left, best, right)]
        ys = [entry.score for entry in (left, best, right)]
        denominator = (xs[0] - xs[1]) * (xs[0] - xs[2]) * (xs[1] - xs[2])
        curvature = (xs[2] * (ys[1] - ys[0]) + xs[1] * (ys[0] - ys[2]) + xs[0] * (ys[2] - ys[1])) / denominator
        slope = (
            xs[2] ** 2 * (ys[0] - ys[1]) + xs[1] ** 2 * (ys[2] - ys[0]) + xs[0] ** 2 * (ys[1] - ys[2])
        ) / denominator
        if curvature < 0:
            vertex = -slope / (2 * curvature)
            if xs[0] < vertex < xs[2]:
                candidate = math.exp(vertex)
                if _admissible(candidate, summaries):
                    return candidate
    neighbours = [entry for entry in (left, right) if entry is not None]
    neighbours.sort(key=lambda entry: entry.score, reverse=True)
    for neighbour in neighbours:
        candidate = math.sqrt(best.ratio * neighbour.ratio)
        if _admissible(candidate, summaries):
            return candidate
    return None


def next_ratio(config: SearchConfig, trials: Sequence[TrialMeasurement]) -> float | None:
    """The ratio to try next, or None when the search is finished."""
    if len(trials) >= config.max_trials:
        return None
    summaries = _summaries(config, trials)
    anchor = _anchor(config)
    for seed in (anchor, 0.0, _clamp(config, anchor * _EXPANSION_FACTOR), _clamp(config, anchor / _EXPANSION_FACTOR)):
        if _admissible(seed, summaries):
            return seed
    if config.max_trials - len(trials) == 1:
        ranked = sorted(summaries, key=lambda summary: summary.score, reverse=True)
        if len(ranked) >= 2 and _overlap(ranked[0], ranked[1]):
            return ranked[0].ratio
    return _refine(config, summaries)


def _fitted_candidate(config: SearchConfig, summaries: Sequence[CandidateSummary]) -> CandidateSummary | None:
    """The candidate measured at the fitted ratio. A fit at or below
    `min_ratio` is the linear model, R = 0."""
    if config.fitted_ratio <= config.min_ratio:
        return next((summary for summary in summaries if summary.ratio == 0), None)
    anchor = _anchor(config)
    for summary in summaries:
        if summary.ratio > 0 and max(anchor, summary.ratio) / min(anchor, summary.ratio) < _EXCLUSION_FACTOR:
            return summary
    return None


def _significant(config: SearchConfig, winner: CandidateSummary, other: CandidateSummary) -> bool:
    if config.objective == "latency":
        return (other.ttft_p90 - winner.ttft_p90) > config.tolerance * other.ttft_p90
    gain = winner.output_tokens_per_second - other.output_tokens_per_second
    return gain > config.tolerance * other.output_tokens_per_second and not _overlap(winner, other)


def _comparison(config: SearchConfig, winner: CandidateSummary, other: CandidateSummary) -> str:
    if config.objective == "latency":
        change = (other.ttft_p90 - winner.ttft_p90) / other.ttft_p90 * 100
        return f"{change:.1f}% lower p90 TTFT than R={format_ratio(other.ratio)}"
    change = (winner.output_tokens_per_second / other.output_tokens_per_second - 1) * 100
    return f"{change:.1f}% more output throughput than R={format_ratio(other.ratio)}"


def select(config: SearchConfig, trials: Sequence[TrialMeasurement]) -> Selection:
    """Rank the measured ratios and choose the one to recommend."""
    summaries = _summaries(config, trials)
    if not summaries:
        raise ValueError("No finished trials to select from.")
    ranked = sorted(summaries, key=lambda summary: summary.score, reverse=True)
    best_score = ranked[0].score
    band = [summary for summary in ranked if summary.score >= best_score - config.tolerance * abs(best_score)]
    winner = min(band, key=lambda summary: summary.ttft_p90)
    ranked.remove(winner)
    ranked.insert(0, winner)

    notes = []
    if config.max_ttft_p90 is not None and not any(summary.meets_slo for summary in summaries):
        notes.append(f"No ratio met the p90 TTFT target of {config.max_ttft_p90:g} s.")

    fitted = _fitted_candidate(config, summaries)
    runner_up = ranked[1] if len(ranked) > 1 else None
    if fitted is not None and fitted is not winner:
        if _significant(config, winner, fitted):
            reason = (
                f"R={format_ratio(winner.ratio)} gave {_comparison(config, winner, fitted)} (95% CIs do not overlap)."
            )
            return Selection(winner.ratio, tuple(ranked), reason, True, " ".join(notes) or None)
        reason = (
            f"No ratio beat the fitted R={format_ratio(fitted.ratio)} beyond measurement noise, "
            "so the fitted ratio is recommended."
        )
        notes.insert(0, f"Best measured: R={format_ratio(winner.ratio)}, {_comparison(config, winner, fitted)}.")
        ranked.remove(fitted)
        ranked.insert(0, fitted)
        return Selection(fitted.ratio, tuple(ranked), reason, False, " ".join(notes))
    if runner_up is None:
        reason = f"Only R={format_ratio(winner.ratio)} was measured."
        return Selection(winner.ratio, tuple(ranked), reason, False, " ".join(notes) or None)
    confident = _significant(config, winner, runner_up)
    reason = f"R={format_ratio(winner.ratio)} gave {_comparison(config, winner, runner_up)}" + (
        " (95% CIs do not overlap)." if confident else ", within measurement noise."
    )
    return Selection(winner.ratio, tuple(ranked), reason, confident, " ".join(notes) or None)
