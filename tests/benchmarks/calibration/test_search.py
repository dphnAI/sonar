# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest

from sonar.benchmarks.calibration.search import SearchConfig, TrialMeasurement, next_ratio, select

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def trial(ratio: float, throughput: float, *, p90: float = 1.0, spread: float = 1.0, failed: int = 0):
    intervals = tuple(throughput + spread * offset for offset in (-1.0, 0.0, 1.0, 0.0))
    return TrialMeasurement(
        ratio=ratio,
        output_tokens_per_second=throughput,
        interval_output_tokens_per_second=intervals,
        ttft_p50=p90 / 2,
        ttft_p90=p90,
        completed_requests=1000,
        failed_requests=failed,
        prefix_hit_rate=0.9,
        frontier_hit_tokens=0,
        kv_usage_p50=0.98,
    )


def config(**overrides) -> SearchConfig:
    values = {"fitted_ratio": 0.2, "fitted_interval": (0.1, 0.3), "max_trials": 8}
    return SearchConfig(**(values | overrides))


def run_search(cfg: SearchConfig, score) -> list[TrialMeasurement]:
    trials: list[TrialMeasurement] = []
    while (ratio := next_ratio(cfg, trials)) is not None:
        trials.append(trial(ratio, score(ratio)))
    return trials


def test_seeds_start_at_the_fitted_ratio_then_bracket_it():
    cfg = config()
    trials: list[TrialMeasurement] = []
    ratios = []
    for _ in range(4):
        ratio = next_ratio(cfg, trials)
        ratios.append(ratio)
        trials.append(trial(ratio, 100.0))
    assert ratios == pytest.approx([0.2, 0.0, 0.8, 0.05])


def test_tiny_fitted_ratio_uses_the_default_anchor_and_clamps():
    cfg = config(fitted_ratio=0.0, max_ratio=0.5)
    first = next_ratio(cfg, [])
    assert first == pytest.approx(0.2)
    trials = [trial(0.2, 1.0), trial(0.0, 1.0)]
    assert next_ratio(cfg, trials) == pytest.approx(0.5)


def test_expands_outward_when_the_best_is_at_an_edge():
    cfg = config()
    trials = run_search(config(max_trials=4), lambda r: 100 + 10 * math.log(r + 1e-9) if r else 50)
    assert next_ratio(cfg, trials) == pytest.approx(3.2)


def test_refines_inside_the_bracket_around_a_peak():
    peak = 0.35
    cfg = config(max_trials=6)
    trials = run_search(cfg, lambda r: 1000 - 50 * math.log(r / peak) ** 2 if r else 900)
    refined = [t.ratio for t in trials[4:]]
    assert refined, "the search should refine after the seeds"
    assert all(0.05 < ratio < 0.8 for ratio in refined)
    assert min(abs(math.log(ratio / peak)) for ratio in refined) < abs(math.log(0.2 / peak))


def test_never_retries_within_the_exclusion_factor_except_the_final_replicate():
    cfg = config(max_trials=12)
    trials = run_search(cfg, lambda r: 100.0)
    positive = sorted({t.ratio for t in trials if t.ratio > 0})
    for a, b in zip(positive, positive[1:]):
        assert b / a >= 1.25


def test_replicates_the_best_when_the_last_trial_cannot_separate_the_top_two():
    cfg = config(max_trials=5)
    trials = [trial(0.2, 100.0, spread=20), trial(0.0, 99.0, spread=20), trial(0.8, 50.0), trial(0.05, 40.0)]
    assert next_ratio(cfg, trials) == pytest.approx(0.2)


def test_stops_at_the_trial_budget():
    cfg = config(max_trials=3)
    trials = [trial(0.2, 1.0), trial(0.0, 1.0), trial(0.8, 1.0)]
    assert next_ratio(cfg, trials) is None


def test_near_ties_go_to_the_lower_p90_ttft():
    cfg = config(fitted_ratio=0.0)
    trials = [trial(0.2, 1000.0, p90=2.0, spread=0.1), trial(0.8, 1010.0, p90=3.0, spread=0.1)]
    assert select(cfg, trials).ratio == pytest.approx(0.2)


def test_ttft_target_penalises_slow_ratios():
    cfg = config(fitted_ratio=0.0, max_ttft_p90=1.0)
    trials = [trial(0.2, 900.0, p90=0.9, spread=0.1), trial(0.8, 1000.0, p90=2.0, spread=0.1)]
    assert select(cfg, trials).ratio == pytest.approx(0.2)


def test_failed_requests_disqualify_a_ratio():
    cfg = config(fitted_ratio=0.0)
    trials = [trial(0.2, 900.0, spread=0.1), trial(0.8, 2000.0, spread=0.1, failed=50)]
    assert select(cfg, trials).ratio == pytest.approx(0.2)


def test_falls_back_to_the_fitted_ratio_when_the_winner_is_within_noise():
    cfg = config()
    trials = [trial(0.2, 1000.0, spread=50), trial(0.8, 1030.0, spread=50), trial(0.0, 900.0, spread=50)]
    selection = select(cfg, trials)
    assert selection.ratio == pytest.approx(0.2)
    assert not selection.confident
    assert "R=0.8" in (selection.note or "")


def test_recommends_a_clearly_better_ratio_over_the_fit():
    cfg = config()
    trials = [trial(0.2, 1000.0, spread=1), trial(0.8, 1100.0, spread=1), trial(0.0, 900.0, spread=1)]
    selection = select(cfg, trials)
    assert selection.ratio == pytest.approx(0.8)
    assert selection.confident
    assert selection.reason.startswith("R=0.8 gave 10.0% more output throughput than R=0.2")


def test_a_linear_fit_is_compared_against_r_zero():
    cfg = config(fitted_ratio=0.0)
    trials = [trial(0.2, 1030.0, spread=50), trial(0.0, 1000.0, spread=50)]
    selection = select(cfg, trials)
    assert selection.ratio == 0.0
    assert "fitted R=0 " in selection.reason


def test_repeated_ratios_are_pooled():
    cfg = config()
    selection = select(cfg, [trial(0.2, 1000.0), trial(0.2, 1010.0)])
    assert len(selection.candidates) == 1
    assert selection.candidates[0].trials == 2
    assert selection.candidates[0].output_tokens_per_second == pytest.approx(1005.0)
