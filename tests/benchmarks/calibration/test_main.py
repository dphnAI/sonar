# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest
from rich.console import Console

from sonar.benchmarks.calibration import pressure
from sonar.benchmarks.calibration.client import RunnerSnapshot
from sonar.benchmarks.calibration.fit import PrefillSample, fit_prefill_cost
from sonar.benchmarks.calibration.main import (
    CalibrationError,
    estimate_seconds,
    fingerprint,
    inspect_serve_args,
    pressure_notes,
)
from sonar.benchmarks.calibration.options import CalibrationOptions
from sonar.benchmarks.calibration.profile import ProfileMeasurement, prefill_seconds, profile_lengths
from sonar.benchmarks.calibration.search import SearchConfig, TrialMeasurement, select
from sonar.benchmarks.calibration.server import parse_prometheus, replaced_options, rewrite_serve_args
from sonar.benchmarks.calibration.state import CalibrationState, TrialRecord, fingerprint_differences
from sonar.benchmarks.calibration.ui import Dashboard, Plan, ServerView, TrialView, render, render_plan
from sonar.entrypoints.cli.calibrate import split_passthrough

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

FLASHSTATES = [
    "my/model",
    "--max-model-len",
    "65536",
    "--enable-prefix-caching",
    "--mamba-frontier-checkpoint",
    "--prefix-cache-eviction",
    "gds",
]


def test_passthrough_is_split_at_the_first_double_dash():
    argv = ["calibrate", "flashstates", "--effort", "quick", "--", "m", "--port=1", "--", "--x_y", "z"]
    own, serve = split_passthrough(argv)
    assert own == ["calibrate", "flashstates", "--effort", "quick"]
    assert serve == ["m", "--port=1", "--", "--x_y", "z"]
    assert split_passthrough(["serve", "--", "x"]) == (["serve", "--", "x"], None)


def test_managed_options_are_replaced_in_every_spelling():
    serve = ["m", "--host", "0.0.0.0", "--port=9000", "--prefill_cost_coefficients", "1", "0", "--seed", "3"]
    rewritten = rewrite_serve_args(serve, host="127.0.0.1", port=1234, coefficients=(2e-5, 4e-6))
    assert rewritten == [
        "m",
        "--seed",
        "3",
        "--host",
        "127.0.0.1",
        "--port",
        "1234",
        "--prefill-cost-coefficients",
        "2e-05",
        "4e-06",
    ]
    assert replaced_options(serve) == ["--host", "--port", "--prefill-cost-coefficients"]


def test_serve_args_must_enable_flashstates_eviction():
    info = inspect_serve_args(FLASHSTATES + ["--tensor-parallel-size", "2", "--data-parallel-size", "2"])
    assert info.model == "my/model" and info.max_model_len == 65536
    assert info.gpus == 4 and info.data_parallel_size == 2
    with pytest.raises(CalibrationError, match="--prefix-cache-eviction gds"):
        inspect_serve_args(FLASHSTATES[:-2])
    with pytest.raises(CalibrationError, match="--mamba-frontier-checkpoint"):
        inspect_serve_args([arg for arg in FLASHSTATES if arg != "--mamba-frontier-checkpoint"])
    with pytest.raises(CalibrationError, match="Remove --disable-log-stats"):
        inspect_serve_args(FLASHSTATES + ["--disable-log-stats"])
    with pytest.raises(CalibrationError, match="after '--'"):
        inspect_serve_args([])


def test_effort_sets_defaults_that_options_override():
    assert CalibrationOptions(effort="quick").schedule == (300.0, 600.0, 4)
    assert CalibrationOptions(effort="thorough", trials=3, warmup=60).schedule == (60, 1800.0, 3)


def test_pressure_probe_keeps_the_largest_count_that_still_reuses_the_cache():
    assert pressure.ladder(1_000_000, 50_000) == [10, 20, 30, 40]
    assert pressure.ladder(1_000, 50_000) == [2]
    probes = [
        pressure.PressureStep(10, 0.80, 100.0, 1.0),
        pressure.PressureStep(20, 0.70, 140.0, 1.5),
        pressure.PressureStep(30, 0.40, 150.0, 4.0),
    ]
    # 30 sessions fell below 60% of the best hit rate, so 20 is kept.
    sessions, reason = pressure.choose_sessions(probes)
    assert sessions == 20 and "20 sessions kept 70%" in reason
    assert pressure.next_sessions([10, 20, 30, 40], probes) == 40
    # A probe that collapses below 30% of the best ends the ladder.
    collapsed = [*probes[:2], pressure.PressureStep(30, 0.1, 150.0, 9.0)]
    assert pressure.next_sessions([10, 20, 30, 40], collapsed) is None
    # A workload that never reuses the cache falls back to the smallest count, with a warning.
    sessions, reason = pressure.choose_sessions([pressure.PressureStep(10, 0.0, 1.0, 1.0)])
    assert sessions == 10 and "little to decide" in reason


def test_estimate_counts_probes_only_when_sessions_are_automatic():
    schedule = [8192] * 3
    auto = estimate_seconds(CalibrationOptions(effort="quick"), schedule, startup=100)
    fixed = estimate_seconds(CalibrationOptions(effort="quick", sessions=8), schedule, startup=100)
    # Four probes of a fresh server, 150 s warmup and 150 s measurement each.
    assert auto - fixed == pytest.approx(4 * (100 + 150 + 150))


def test_prefill_time_comes_from_the_server_histogram_when_it_saw_one_request():
    before = {"sonar:request_prefill_time_seconds_sum": 10.0, "sonar:request_prefill_time_seconds_count": 4}
    after = {"sonar:request_prefill_time_seconds_sum": 10.75, "sonar:request_prefill_time_seconds_count": 5}
    assert prefill_seconds(before, after, 0.9) == pytest.approx((0.75, "server"))
    two = dict(after, **{"sonar:request_prefill_time_seconds_count": 6})
    assert prefill_seconds(before, two, 0.9) == (0.9, "client")
    assert prefill_seconds({}, {}, 0.9) == (0.9, "client")


def test_profile_lengths_are_geometric_and_bounded():
    lengths = profile_lengths(131072)
    assert lengths[0] == 8192 and lengths[-1] == 131072 and len(lengths) == 8
    # Short limits scale the range down rather than profiling one length.
    assert profile_lengths(16384)[0] == 1024
    with pytest.raises(ValueError, match="at least 4096"):
        profile_lengths(4000)


def test_prometheus_text_is_summed_over_engines_and_skips_buckets():
    text = """
# HELP sonar:prefix_cache_hits_total Prefix cache hits
sonar:prefix_cache_hits_total{engine="0",model_name="m"} 100.0
sonar:prefix_cache_hits_total{engine="1",model_name="m"} 50.0
sonar:request_prefill_time_seconds_bucket{engine="0",le="0.5"} 3.0
sonar:request_prefill_time_seconds_sum{engine="0"} 1.5
sonar:kv_cache_usage_perc{engine="0"} 0.5
"""
    metrics = parse_prometheus(text)
    assert metrics["sonar:prefix_cache_hits_total"] == 150.0
    assert "sonar:request_prefill_time_seconds_bucket" not in metrics
    assert metrics["sonar:request_prefill_time_seconds_sum"] == 1.5


def _measurement(ratio, throughput, usage=0.95, failed=0):
    return TrialMeasurement(ratio, throughput, (throughput,) * 4, 0.5, 1.0, 500, failed, 0.9, 1000, usage)


def _fit():
    samples = [PrefillSample(n, 0.01 + 2e-5 * n + 4e-6 * n * n / 2e5) for n in (1024, 4096, 16384, 65536)]
    return fit_prefill_cost(samples, bootstrap=20)


def test_state_round_trips_and_detects_changed_settings(tmp_path):
    options = CalibrationOptions()
    state = CalibrationState(
        invocation={"options": {}, "serve_args": FLASHSTATES}, fingerprint=fingerprint(options, FLASHSTATES)
    )
    state.profile = [ProfileMeasurement(1024, 0.1, "server")]
    state.fit = _fit()
    state.trials = [TrialRecord(1, (2e-5, 4e-6), 8, "a", "b", "dir", _measurement(0.2, 1000.0))]
    state.save(tmp_path)
    loaded = CalibrationState.load(tmp_path)
    assert loaded.fit == state.fit
    assert loaded.trials == state.trials
    assert loaded.profile == state.profile
    changed = fingerprint(CalibrationOptions(effort="quick"), FLASHSTATES)
    assert fingerprint_differences(loaded.fingerprint, changed) == ["effort: saved 'standard', now 'quick'"]


def test_reports_warn_without_cache_pressure_or_with_failures():
    notes = pressure_notes([_measurement(0.2, 1.0, usage=0.5), _measurement(0.0, 1.0, usage=0.6, failed=3)])
    assert "Increase --sessions" in notes[0]
    assert "3 requests failed" in notes[1]
    assert pressure_notes([_measurement(0.2, 1.0, usage=0.99)]) == []


def test_reports_warn_when_requests_evict_each_other():
    thrashing = TrialMeasurement(0.2, 100.0, (100.0,) * 4, 50.0, 80.0, 70, 0, 0.0, 0, 0.98)
    assert "Lower --sessions" in pressure_notes([thrashing])[0]


def _render(renderable) -> str:
    console = Console(record=True, width=120)
    console.print(renderable)
    return console.export_text()


def test_dashboard_renders_each_stage():
    dashboard = Dashboard(model="my/model", gpus=2, workload="5 synthetic sessions", output_dir="out")
    dashboard.server = ServerView("Starting the profiling server", "Loading weights", time.monotonic(), "log line")
    assert "Loading weights" in _render(render(dashboard))

    dashboard.server = None
    dashboard.profile_total = 6
    dashboard.profile = [ProfileMeasurement(n, 0.01 + n * 2e-5, "server") for n in (1024, 4096, 16384)]
    dashboard.fit = _fit()
    profile = _render(render(dashboard))
    assert "16,384" in profile and "R = E/D" in profile and "3/6 requests" in profile

    dashboard.stage = "pressure"
    dashboard.pressure_counts = [10, 20, 30, 40]
    dashboard.pressure = [pressure.PressureStep(10, 0.8, 1000.0, 1.2)]
    probe = _render(render(dashboard))
    assert "Cache pressure" in probe and "80.0%" in probe and "pending" in probe

    config = SearchConfig(fitted_ratio=dashboard.fit.ratio, fitted_interval=(0.1, 0.3), max_trials=6)
    dashboard.stage = "search"
    dashboard.search_config = config
    dashboard.trials = [_measurement(dashboard.fit.ratio, 2000.0), _measurement(0.0, 1500.0)]
    snapshot = RunnerSnapshot(30.0, 7, 3, 120, 1, 1800.0, 0.8, 9)
    dashboard.trial = TrialView(
        3, 6, 0.8, 8, 60.0, 120.0, "running", time.monotonic() - 70, lambda: snapshot, 0.9, 0.97
    )
    search = _render(render(dashboard))
    assert "Trial 3/6" in search and "measuring" in search and "900 tok/s/GPU" in search
    assert "★" in search and "running" in search

    dashboard.trial = None
    dashboard.selection = select(config, dashboard.trials)
    dashboard.recommended = dashboard.fit.coefficients(dashboard.selection.ratio)
    dashboard.stage = "report"
    report = _render(render(dashboard))
    assert "--prefill-cost-coefficients" in report and "Recommended" in report


def test_plan_lists_the_schedule_and_managed_flags():
    plan = Plan(
        model="my/model",
        gpus=2,
        workload="393 coding-agent sessions",
        workload_details="10 requests",
        profile_requests=30,
        profile_min_tokens=8192,
        profile_max_tokens=131072,
        trials=6,
        warmup=600,
        duration=1200,
        sessions="automatic, from the KV cache size",
        objective="maximise output throughput",
        estimated_seconds=4 * 3600,
        output_dir="out",
        replaced=("--port",),
    )
    text = _render(render_plan(plan))
    assert "4:00:00" in text and "131,072" in text and "replacing your --port" in text
