# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`sonar calibrate flashstates`: find the prefill cost coefficients that
serve a workload best."""

import argparse
import asyncio
import contextlib
import dataclasses
import datetime
import json
import math
import os
import signal
import statistics
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from rich.console import Console
from rich.prompt import Confirm

from sonar.benchmarks.calibration import pressure
from sonar.benchmarks.calibration.client import WorkloadRunner, summarize_window
from sonar.benchmarks.calibration.fit import FitError, fit_prefill_cost
from sonar.benchmarks.calibration.options import CalibrationOptions, options_from_args
from sonar.benchmarks.calibration.profile import ProfileMeasurement, profile_schedule, run_profile
from sonar.benchmarks.calibration.search import SearchConfig, Selection, TrialMeasurement, format_ratio, next_ratio
from sonar.benchmarks.calibration.search import select as select_ratio
from sonar.benchmarks.calibration.server import ManagedServer, ServerStartError, replaced_options
from sonar.benchmarks.calibration.state import (
    CalibrationState,
    TrialRecord,
    fingerprint_differences,
)
from sonar.benchmarks.calibration.ui import (
    Dashboard,
    LiveReporter,
    PlainReporter,
    Plan,
    ServerView,
    TrialView,
    format_coefficients,
    format_duration,
    render,
    render_plan,
)
from sonar.benchmarks.calibration.workload import TokenSampler, Workload, WorkloadError, load_workload

DEFAULT_STARTUP_ESTIMATE = 300.0
METRICS_INTERVAL = 10.0
# Calibration options that change the result; the others only affect presentation.
_FINGERPRINT_OPTIONS = (
    "workload",
    "sessions",
    "trials",
    "effort",
    "warmup",
    "duration",
    "objective",
    "max_ttft_p90",
    "profile_max_tokens",
    "seed",
)


class CalibrationError(RuntimeError):
    """A problem the user can fix; the message says how."""


# Serve arguments


@dataclass(frozen=True, slots=True)
class ServeInfo:
    model: str
    gpus: int
    data_parallel_size: int
    max_model_len: int | None
    tokenizer: str
    tokenizer_revision: str | None
    trust_remote_code: bool


def inspect_serve_args(serve_args: Sequence[str]) -> ServeInfo:
    """Parse the user's serve arguments with Sonar's own parser and check
    that calibration applies to them."""
    if not serve_args:
        raise CalibrationError(
            "Pass your serve command after '--', for example:\n"
            "  sonar calibrate flashstates -- MODEL --enable-prefix-caching --mamba-frontier-checkpoint "
            "--prefix-cache-eviction gds --prefix-cache-leaf-last"
        )
    from sonar.entrypoints.common.cli_args import make_arg_parser, validate_parsed_serve_args
    from sonar.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(prog="sonar serve", exit_on_error=False))
    try:
        args = parser.parse_args(list(serve_args))
        validate_parsed_serve_args(args)
    except (argparse.ArgumentError, ValueError, TypeError) as error:
        raise CalibrationError(f"The serve arguments are invalid: {error}") from None
    except SystemExit:
        raise CalibrationError("The serve arguments are invalid (see the message above).") from None
    model = args.model_tag or args.model
    missing = []
    if not args.mamba_frontier_checkpoint:
        missing.append("--mamba-frontier-checkpoint")
    if args.prefix_cache_eviction != "gds":
        missing.append("--prefix-cache-eviction gds")
    if missing:
        raise CalibrationError(
            "The cost coefficients only affect FlashStates GreedyDual-Size eviction. Add "
            + " and ".join(missing)
            + " to the serve arguments."
        )
    if args.disable_log_stats:
        raise CalibrationError("Calibration reads the server's metrics. Remove --disable-log-stats.")
    data_parallel = args.data_parallel_size or 1
    gpus = (
        args.tensor_parallel_size
        * args.pipeline_parallel_size
        * data_parallel
        * (getattr(args, "prefill_context_parallel_size", 1) or 1)
    )
    max_model_len = args.max_model_len if isinstance(args.max_model_len, int) and args.max_model_len > 0 else None
    if max_model_len is None:
        max_model_len = _config_max_model_len(model, args.revision, args.trust_remote_code)
    return ServeInfo(
        model=model,
        gpus=gpus,
        data_parallel_size=data_parallel,
        max_model_len=max_model_len,
        tokenizer=args.tokenizer or model,
        tokenizer_revision=args.tokenizer_revision or args.revision,
        trust_remote_code=bool(args.trust_remote_code),
    )


def _config_max_model_len(model: str, revision: str | None, trust_remote_code: bool) -> int | None:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
    except Exception:
        return None
    for candidate in (config, getattr(config, "text_config", None)):
        value = getattr(candidate, "max_position_embeddings", None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def load_sampler(info: ServeInfo, seed: int) -> TokenSampler:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        info.tokenizer, revision=info.tokenizer_revision, trust_remote_code=info.trust_remote_code
    )
    excluded = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    return TokenSampler(len(tokenizer), excluded, seed)


def default_profile_max_tokens(max_model_len: int, workload: Workload) -> int:
    return min(max_model_len - 1, max(16384, workload.stats.prompt_tokens_p95))


def estimate_seconds(
    options: CalibrationOptions, schedule: Sequence[int], startup: float, prefill_tokens_per_second: float = 20000.0
) -> float:
    warmup, duration, trials = options.schedule
    profile = sum(schedule) / prefill_tokens_per_second + 2 * len(schedule)
    probe = 0.0
    if options.sessions is None:
        probe_warmup, probe_duration = options.probe_schedule
        probe = len(pressure.MULTIPLIERS) * (startup + probe_warmup + probe_duration)
    return startup + profile + probe + trials * (startup + warmup + duration)


def fingerprint(options: CalibrationOptions, serve_args: Sequence[str]) -> dict[str, Any]:
    import sonar

    data: dict[str, Any] = {name: getattr(options, name) for name in _FINGERPRINT_OPTIONS}
    data["serve_args"] = list(serve_args)
    data["sonar_version"] = getattr(sonar, "__version__", "unknown")
    return data


def default_output_dir(model: str) -> Path:
    name = Path(model.rstrip("/")).name or "model"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("sonar-calibration") / f"{name}-{stamp}"


# Measurement helpers


def _counter(metrics: dict[str, float], name: str) -> float | None:
    for key in (name, f"{name}_total"):
        if key in metrics:
            return metrics[key]
    return None


def _delta(start: dict[str, float], end: dict[str, float], name: str) -> float | None:
    before, after = _counter(start, name), _counter(end, name)
    return None if before is None or after is None else after - before


async def _scrape(server: ManagedServer, http: aiohttp.ClientSession) -> dict[str, float] | None:
    """Server metrics, or None when this scrape failed; the next one retries."""
    try:
        return await server.metrics(http)
    except (TimeoutError, aiohttp.ClientError):
        return None


def trial_measurement(
    ratio: float,
    summary: Any,
    start_metrics: dict[str, float],
    end_metrics: dict[str, float],
    kv_samples: Sequence[float],
) -> TrialMeasurement:
    queries = _delta(start_metrics, end_metrics, "sonar:prefix_cache_queries")
    hits = _delta(start_metrics, end_metrics, "sonar:prefix_cache_hits")
    frontier = _delta(start_metrics, end_metrics, "sonar:prefix_cache_frontier_hits")
    ttft_p90 = summary.ttft_p90 if math.isfinite(summary.ttft_p90) else math.inf
    ttft_p50 = summary.ttft_p50 if math.isfinite(summary.ttft_p50) else math.inf
    return TrialMeasurement(
        ratio=ratio,
        output_tokens_per_second=summary.output_tokens_per_second,
        interval_output_tokens_per_second=summary.interval_output_tokens_per_second,
        ttft_p50=ttft_p50,
        ttft_p90=ttft_p90,
        completed_requests=summary.completed_requests,
        failed_requests=summary.failed_requests,
        prefix_hit_rate=(hits / queries) if hits is not None and queries else None,
        frontier_hit_tokens=int(frontier) if frontier is not None else None,
        kv_usage_p50=statistics.median(kv_samples) if kv_samples else None,
    )


# Reporting


def pressure_notes(trials: Sequence[TrialMeasurement]) -> list[str]:
    """Warnings about the trial conditions. KV cache usage is not one: Sonar
    counts only blocks held by running requests, so a cache full of evictable
    prefixes still reports low usage."""
    notes = []
    hit_rates = [trial.prefix_hit_rate for trial in trials if trial.prefix_hit_rate is not None]
    if hit_rates and max(hit_rates) < 0.1:
        notes.append(
            f"Prefix cache hits stayed at or below {max(hit_rates) * 100:.1f}%: requests evicted each other's "
            "prefixes before reuse, so no eviction order could help. Lower --sessions."
        )
    failed = sum(trial.failed_requests for trial in trials)
    if failed:
        notes.append(f"{failed} requests failed during measurement windows; see the trial server logs.")
    return notes


def write_report(
    output: Path,
    state: CalibrationState,
    config: SearchConfig,
    selection: Selection,
    recommended: tuple[float, float],
    gpus: int,
    workload: Workload,
) -> list[Path]:
    fit = state.fit
    assert fit is not None
    notes = [note for note in (selection.note,) if note] + pressure_notes(state.measurements)
    result = {
        "prefill_cost_coefficients": list(recommended),
        "ratio": selection.ratio,
        "confident": selection.confident,
        "reason": selection.reason,
        "notes": notes,
        "fit": dataclasses.asdict(fit),
        "sessions": state.sessions,
        "workload": workload.description,
        "candidates": [dataclasses.asdict(candidate) for candidate in selection.candidates],
        "trials": [dataclasses.asdict(trial) for trial in state.trials],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# FlashStates calibration",
        "",
        "## Recommendation",
        "",
        "```",
        format_coefficients(recommended),
        "```",
        "",
        selection.reason,
        "",
        *[line for note in notes for line in (f"- {note}", "")],
        "## Prefill cost fit",
        "",
        (
            f"`T(L) = C + D·L + E·L²/200000` over {fit.num_samples} uncached prefills from "
            f"{fit.min_tokens:,} to {fit.max_tokens:,} tokens."
        ),
        "",
        "| D (s/token) | E | C (s) | R = E/D | 90% interval | R² | worst error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {fit.per_token:.4g} | {fit.per_token_per_100k:.4g} | {fit.intercept:.4g} "
            f"| {format_ratio(fit.ratio)} | {fit.ratio_interval[0]:.3g}–{fit.ratio_interval[1]:.3g} "
            f"| {fit.r_squared:.4f} | {fit.max_relative_error * 100:.1f}% |"
        ),
        "",
        "## Trials",
        "",
        f"{workload.description}, {state.sessions} concurrent sessions, {gpus} GPU{'s' if gpus != 1 else ''}."
        + (f" {state.sessions_note}" if state.sessions_note else ""),
        "",
        "| R | Runs | Output tok/s/GPU | 95% CI | p90 TTFT (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for candidate in sorted(selection.candidates, key=lambda c: c.ratio):
        low, high = candidate.ci95
        ci = f"{low / gpus:,.0f}–{high / gpus:,.0f}" if math.isfinite(high) else "-"
        marker = " (recommended)" if candidate.ratio == selection.ratio else ""
        lines.append(
            f"| {format_ratio(candidate.ratio)}{marker} | {candidate.trials} "
            f"| {candidate.output_tokens_per_second / gpus:,.0f} | {ci} | {candidate.ttft_p90:.2f} |"
        )
    lines += ["", "## Serve arguments", "", "```", " ".join(state.invocation["serve_args"]), "```", ""]
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines))
    return [report_path, result_path]


# Orchestration


class Calibration:
    def __init__(
        self,
        options: CalibrationOptions,
        serve_args: list[str],
        info: ServeInfo,
        workload: Workload,
        sampler: TokenSampler,
        state: CalibrationState,
        output: Path,
        dashboard: Dashboard,
        reporter: LiveReporter | PlainReporter,
    ) -> None:
        self.options = options
        self.serve_args = serve_args
        self.info = info
        self.workload = workload
        self.sampler = sampler
        self.state = state
        self.output = output
        self.dashboard = dashboard
        self.reporter = reporter
        self.server: ManagedServer | None = None
        warmup, duration, trials = options.schedule
        self.warmup, self.duration, self.max_trials = warmup, duration, trials
        self.served_model = info.model
        self.server_max_model_len: int | None = None

    # UI plumbing

    async def _ticker(self) -> None:
        while True:
            server, view = self.server, self.dashboard.server
            if server is not None and view is not None:
                view.phase = server.phase
                view.last_line = server.last_line
            self.dashboard.eta_seconds = self._eta()
            self.reporter.tick()
            await asyncio.sleep(0.25)

    def _startup_estimate(self) -> float:
        return statistics.mean(self.state.startup_seconds) if self.state.startup_seconds else DEFAULT_STARTUP_ESTIMATE

    def _eta(self) -> float | None:
        startup = self._startup_estimate()
        trial = self.dashboard.trial
        running = None
        if trial is not None and trial.run_started is not None:
            running = max(0.0, trial.warmup + trial.duration - (time.monotonic() - trial.run_started))
        remaining = 0.0
        if self.state.fit is None:
            done = self.state.profile
            schedule = self.state.profile_schedule or []
            mean = statistics.mean(m.seconds for m in done) + 1.0 if done else 5.0
            remaining += mean * (len(schedule) - len(done)) + (startup if self.server is None else 0.0)
        if self.state.sessions is None and self.options.sessions is None:
            probe_warmup, probe_duration = self.options.probe_schedule
            per_probe = startup + probe_warmup + probe_duration
            probes_left = len(self.dashboard.pressure_counts or pressure.MULTIPLIERS) - len(self.state.pressure)
            if self.dashboard.stage == "pressure" and trial is not None:
                probes_left -= 1
                remaining += running if running is not None else per_probe
            remaining += probes_left * per_probe
        per_trial = startup + self.warmup + self.duration
        trials_left = self.max_trials - len(self.state.trials)
        if self.dashboard.stage == "search" and trial is not None:
            trials_left -= 1
            remaining += running if running is not None else per_trial
        return remaining + max(0, trials_left) * per_trial

    def _save(self) -> None:
        self.state.save(self.output)

    async def _start_server(self, label: str, coefficients: tuple[float, float], log_path: Path) -> ManagedServer:
        server = ManagedServer(self.serve_args, coefficients, log_path, startup_timeout=self.options.startup_timeout)
        self.server = server
        self.dashboard.server = ServerView(label=label, phase=server.phase, started=time.monotonic())
        self.reporter.event(f"{label}: launching sonar serve (log: {log_path})")
        server.start()
        try:
            await server.wait_ready()
        except BaseException:
            await asyncio.to_thread(server.stop)
            self.server = None
            self.dashboard.server = None
            raise
        assert server.startup_seconds is not None
        self.state.startup_seconds.append(server.startup_seconds)
        async with aiohttp.ClientSession() as http:
            self.served_model, self.server_max_model_len = await server.model_info(http)
        self.dashboard.server = None
        self.reporter.event(f"Server ready after {format_duration(server.startup_seconds)}")
        return server

    async def _stop_server(self) -> None:
        server, self.server = self.server, None
        if server is not None:
            await asyncio.to_thread(server.stop)

    # Stages

    async def run(self) -> None:
        ticker = asyncio.create_task(self._ticker())
        try:
            if self.state.fit is None:
                await self._profile()
            self.dashboard.fit = self.state.fit
            await self._pressure()
            self.dashboard.sessions_note = self.state.sessions_note
            await self._search()
            self._report()
        finally:
            ticker.cancel()
            await self._stop_server()

    async def _profile(self) -> None:
        self.dashboard.stage = "profile"
        self.dashboard.profile = list(self.state.profile)
        self.dashboard.profile_total = len(self.state.profile_schedule or [])
        server = await self._start_server(
            "Starting the profiling server", (1.0, 0.0), self.output / "profile" / "server.log"
        )
        server_max_len = self.server_max_model_len
        if server_max_len and server_max_len != self.info.max_model_len:
            self.reporter.event(f"The server uses max_model_len {server_max_len:,}")
        self.state.max_model_len = server_max_len or self.info.max_model_len
        self.state.kv_capacity_tokens = server.kv_capacity_tokens
        self._save()

        def on_measurement(measurement: ProfileMeasurement) -> None:
            self.dashboard.profile.append(measurement)
            self._save()
            samples = [m.sample() for m in self.state.profile]
            if len({s.tokens for s in samples}) >= 3:
                with contextlib.suppress(FitError):
                    self.dashboard.fit = fit_prefill_cost(samples, bootstrap=0)

        def on_status(status: str) -> None:
            self.dashboard.profile_status = status

        assert self.state.profile_schedule is not None
        await run_profile(
            server,
            self.served_model,
            self.sampler,
            self.state.profile_schedule,
            self.state.profile,
            on_measurement=on_measurement,
            on_status=on_status,
        )
        await self._stop_server()
        try:
            fit = fit_prefill_cost([m.sample() for m in self.state.profile], seed=self.options.seed)
        except FitError as error:
            raise CalibrationError(str(error)) from None
        clients = sum(m.source == "client" for m in self.state.profile)
        if clients:
            self.reporter.event(f"{clients} prefill timings came from client TTFT; the server histogram missed them")
        self.state.fit = fit
        self.dashboard.fit = fit
        self._save()
        self.reporter.event(
            f"Fitted D={fit.per_token:.4g} E={fit.per_token_per_100k:.4g} (R={format_ratio(fit.ratio)}, "
            f"R² {fit.r_squared:.4f})"
        )

    def _search_config(self) -> SearchConfig:
        fit = self.state.fit
        assert fit is not None
        return SearchConfig(
            fitted_ratio=fit.ratio,
            fitted_interval=fit.ratio_interval,
            max_trials=self.max_trials,
            objective=self.options.objective,  # type: ignore[arg-type]
            max_ttft_p90=self.options.max_ttft_p90,
        )

    async def _search(self) -> None:
        self.dashboard.stage = "search"
        config = self._search_config()
        self.dashboard.search_config = config
        self.dashboard.trials = list(self.state.measurements)
        sessions = self.state.sessions
        assert sessions is not None
        while (ratio := next_ratio(config, self.state.measurements)) is not None:
            await self._trial(len(self.state.trials) + 1, ratio, config, sessions)

    async def _run_workload(
        self, server: ManagedServer, view: TrialView, lanes: int, warmup: float, duration: float
    ) -> tuple[Any, dict[str, float], dict[str, float], list[float]]:
        """Replay the workload with `lanes` sessions for `warmup + duration`
        seconds, tracking server metrics, and summarise the measurement window."""
        max_model_len = self.state.max_model_len or self.info.max_model_len
        assert max_model_len is not None
        runner = WorkloadRunner(
            server.base_url,
            self.served_model,
            self.workload,
            lanes=lanes,
            sampler=self.sampler,
            max_model_len=max_model_len,
            seed=self.options.seed,
        )
        view.snapshot = runner.snapshot
        view.phase = "running"
        total = warmup + duration
        view.run_started = time.monotonic()
        run = asyncio.create_task(runner.run(total))
        window_start = view.run_started + warmup
        window_end = view.run_started + total
        start_metrics: dict[str, float] | None = None
        kv_samples: list[float] = []
        async with aiohttp.ClientSession() as http:
            while not run.done():
                if not server.alive():
                    run.cancel()
                    raise CalibrationError(
                        "The server exited unexpectedly. Last log lines:\n" + "\n".join(server.tail()[-30:])
                    )
                now = time.monotonic()
                metrics = await _scrape(server, http)
                if start_metrics is None and now >= window_start:
                    start_metrics = metrics
                usage = metrics.get("sonar:kv_cache_usage_perc") if metrics else None
                if usage is not None:
                    view.kv_usage = usage / self.info.data_parallel_size
                    if window_start <= now < window_end:
                        kv_samples.append(view.kv_usage)
                if metrics:
                    # Hit rate over the window once it starts; before that, since the run began.
                    base = start_metrics or view.baseline_metrics
                    if base is None:
                        view.baseline_metrics = metrics
                    else:
                        queries = _delta(base, metrics, "sonar:prefix_cache_queries")
                        hits = _delta(base, metrics, "sonar:prefix_cache_hits")
                        if queries:
                            view.prefix_hit_rate = (hits or 0.0) / queries
                next_tick = min(now + METRICS_INTERVAL, window_start if start_metrics is None else math.inf)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(run), timeout=max(0.0, next_tick - time.monotonic()))
            records = run.result()
            end_metrics = await _scrape(server, http) or {}
        # At least ten throughput samples per window for the confidence interval.
        interval = min(60.0, max(5.0, duration / 10))
        summary = summarize_window(records, runner.timeline, window_start, window_end, interval=interval)
        return summary, start_metrics or end_metrics, end_metrics, kv_samples

    async def _pressure(self) -> None:
        """Choose the session count, probing the cache unless --sessions is set."""
        if self.state.sessions is not None:
            return
        if self.options.sessions is not None:
            self.state.sessions = self.options.sessions
            self._save()
            return
        capacity = self.state.kv_capacity_tokens
        if capacity is None:
            raise CalibrationError(
                "The server did not report its KV cache size, so the session count cannot be probed. Pass --sessions."
            )
        capacity *= self.info.data_parallel_size
        counts = pressure.ladder(capacity, self.workload.stats.mean_session_context)
        self.dashboard.stage = "pressure"
        self.dashboard.pressure = list(self.state.pressure)
        self.dashboard.pressure_counts = counts
        fit = self.state.fit
        assert fit is not None
        warmup, duration = self.options.probe_schedule
        while (lanes := pressure.next_sessions(counts, self.state.pressure)) is not None:
            index = len(self.state.pressure) + 1
            view = TrialView(
                index=index,
                total=len(counts),
                ratio=fit.ratio,
                sessions=lanes,
                warmup=warmup,
                duration=duration,
                title=f"Cache-pressure probe {index}/{len(counts)} · {lanes} sessions",
            )
            self.dashboard.trial = view
            # A fresh server per probe, so each starts from an empty cache.
            server = await self._start_server(
                f"Starting the server for probe {index} ({lanes} sessions)",
                fit.coefficients(fit.ratio),
                self.output / "pressure" / f"{index:02d}-{lanes}-sessions" / "server.log",
            )
            load = lanes * self.workload.stats.mean_session_context / capacity
            self.reporter.event(f"Probing {lanes} sessions ({load:.1f}x the KV cache)")
            summary, start, end, _ = await self._run_workload(server, view, lanes, warmup, duration)
            await self._stop_server()
            queries = _delta(start, end, "sonar:prefix_cache_queries")
            hits = _delta(start, end, "sonar:prefix_cache_hits")
            step = pressure.PressureStep(
                sessions=lanes,
                prefix_hit_rate=(hits / queries) if hits is not None and queries else None,
                output_tokens_per_second=summary.output_tokens_per_second,
                ttft_p90=summary.ttft_p90 if math.isfinite(summary.ttft_p90) else math.inf,
            )
            self.state.pressure.append(step)
            self._save()
            self.dashboard.pressure.append(step)
            self.dashboard.trial = None
            rate = "-" if step.prefix_hit_rate is None else f"{step.prefix_hit_rate * 100:.1f}%"
            self.reporter.event(
                f"{lanes} sessions: {rate} prefix hits, "
                f"{step.output_tokens_per_second / self.info.gpus:,.0f} output tok/s/GPU, "
                f"p90 TTFT {step.ttft_p90:.2f} s"
            )
        sessions, note = pressure.choose_sessions(self.state.pressure)
        self.state.sessions = sessions
        self.state.sessions_note = note
        self._save()
        self.dashboard.sessions_note = note
        self.reporter.event(note)

    async def _trial(self, index: int, ratio: float, config: SearchConfig, sessions: int) -> None:
        fit = self.state.fit
        assert fit is not None
        coefficients = fit.coefficients(ratio)
        directory = self.output / "trials" / f"{index:02d}-R{format_ratio(ratio)}"
        trial = TrialView(
            index=index,
            total=config.max_trials,
            ratio=ratio,
            sessions=sessions,
            warmup=self.warmup,
            duration=self.duration,
            title=f"Trial {index}/{config.max_trials} · R = {format_ratio(ratio)}",
        )
        self.dashboard.trial = trial
        started = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        server = await self._start_server(
            f"Starting the server for trial {index} (R = {format_ratio(ratio)})",
            coefficients,
            directory / "server.log",
        )
        self.reporter.event(
            f"Trial {index}/{config.max_trials}: R = {format_ratio(ratio)} "
            f"({format_coefficients(coefficients)}), {sessions} sessions"
        )
        summary, start, end, kv_samples = await self._run_workload(server, trial, sessions, self.warmup, self.duration)
        trial.phase = "stopping"
        await self._stop_server()
        measurement = trial_measurement(ratio, summary, start, end, kv_samples)
        record = TrialRecord(
            index=index,
            coefficients=coefficients,
            sessions=sessions,
            started=started,
            finished=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            directory=str(directory),
            measurement=measurement,
        )
        (directory / "summary.json").write_text(json.dumps(dataclasses.asdict(record), indent=2) + "\n")
        self.state.trials.append(record)
        self._save()
        self.dashboard.trials.append(measurement)
        self.dashboard.trial = None
        hit_note = (
            f", {measurement.prefix_hit_rate * 100:.1f}% prefix hits" if measurement.prefix_hit_rate is not None else ""
        )
        self.reporter.event(
            f"Trial {index} R = {format_ratio(ratio)}: "
            f"{measurement.output_tokens_per_second / self.info.gpus:,.0f} output tok/s/GPU, "
            f"p90 TTFT {measurement.ttft_p90:.2f} s{hit_note}"
        )

    def _report(self) -> None:
        fit = self.state.fit
        assert fit is not None
        config = self._search_config()
        selection = select_ratio(config, self.state.measurements)
        recommended = fit.coefficients(selection.ratio)
        files = write_report(self.output, self.state, config, selection, recommended, self.info.gpus, self.workload)
        self.state.finished = True
        self._save()
        self.dashboard.selection = selection
        self.dashboard.recommended = recommended
        self.dashboard.files = [str(path) for path in files]
        self.dashboard.stage = "report"
        self.dashboard.eta_seconds = 0.0


# Entry point


def _load_workload(options: CalibrationOptions, max_model_len: int, console: Console) -> Workload:
    with console.status("Loading the workload…") as status:

        def progress(message: str, fraction: float | None) -> None:
            status.update(f"{message}…" if fraction is None else f"{message} {fraction * 100:.0f}%")

        try:
            return load_workload(options.workload, max_model_len=max_model_len, seed=options.seed, progress=progress)
        except WorkloadError as error:
            raise CalibrationError(str(error)) from None


def _prepare_state(
    options: CalibrationOptions, serve_args: list[str], resume: Path | None
) -> tuple[CalibrationState, list[str], CalibrationOptions]:
    if resume is None:
        state = CalibrationState(
            invocation={"options": dataclasses.asdict(options), "serve_args": serve_args},
            fingerprint=fingerprint(options, serve_args),
        )
        return state, serve_args, options
    try:
        state = CalibrationState.load(resume)
    except (FileNotFoundError, ValueError) as error:
        raise CalibrationError(str(error)) from None
    saved_options = CalibrationOptions(**state.invocation["options"])
    if serve_args:
        differences = fingerprint_differences(state.fingerprint, fingerprint(options, serve_args))
        if differences:
            raise CalibrationError(
                "This calibration was started with different settings, so its results cannot be reused:\n  "
                + "\n  ".join(differences)
                + f"\nRun `sonar calibrate flashstates --resume {resume}` without other arguments to continue it."
            )
    saved_options.resume = str(resume)
    saved_options.yes = saved_options.yes or options.yes
    saved_options.no_tui = options.no_tui
    return state, list(state.invocation["serve_args"]), saved_options


def run(args: argparse.Namespace) -> int:
    options = options_from_args(args)
    serve_args = list(getattr(args, "serve_args", None) or [])
    console = Console(stderr=False)
    try:
        return _run(options, serve_args, console)
    except CalibrationError as error:
        console.print(f"[bold red]Calibration stopped:[/] {error}", highlight=False)
        return 2


def _run(options: CalibrationOptions, serve_args: list[str], console: Console) -> int:
    resume = Path(options.resume) if options.resume else None
    state, serve_args, options = _prepare_state(options, serve_args, resume)
    info = inspect_serve_args(serve_args)
    max_model_len = state.max_model_len or info.max_model_len
    if max_model_len is None:
        raise CalibrationError(
            "Cannot determine the model's maximum length. Pass --max-model-len in the serve arguments."
        )
    output = resume or Path(options.output_dir or default_output_dir(info.model))
    workload = _load_workload(options, max_model_len, console)
    with console.status("Loading the tokenizer…"):
        try:
            sampler = load_sampler(info, options.seed)
        except (OSError, ValueError) as error:
            raise CalibrationError(
                f"Could not load the tokenizer of {info.tokenizer!r}: {error}\n"
                "Check that the model is downloaded, or that the Hugging Face Hub is reachable."
            ) from None

    if state.profile_schedule is None:
        max_tokens = options.profile_max_tokens or default_profile_max_tokens(max_model_len, workload)
        if max_tokens > max_model_len - 1:
            raise CalibrationError(f"--profile-max-tokens must be below max_model_len ({max_model_len:,}).")
        try:
            state.profile_schedule = profile_schedule(max_tokens, options.seed)
        except ValueError as error:
            raise CalibrationError(str(error)) from None
    warmup, duration, trials = options.schedule
    if resume is None:
        stats = workload.stats
        details = (
            f"{stats.requests:,} requests, {stats.mean_turns:.0f} turns per session, prompts p50 "
            f"{stats.prompt_tokens_p50:,} / p95 {stats.prompt_tokens_p95:,} tokens"
        )
        if stats.truncated_sessions:
            details += f", {stats.truncated_sessions} sessions cut at max_model_len {max_model_len:,}"
        plan = Plan(
            model=info.model,
            gpus=info.gpus,
            workload=workload.description,
            workload_details=details,
            profile_requests=len(state.profile_schedule),
            profile_min_tokens=min(state.profile_schedule),
            profile_max_tokens=max(state.profile_schedule),
            trials=trials,
            warmup=warmup,
            duration=duration,
            sessions=(
                str(options.sessions)
                if options.sessions
                else "probed: up to 4 short runs around the KV cache capacity on fresh servers"
            ),
            objective=(
                "maximise output throughput"
                if options.objective == "throughput"
                else "minimise p90 TTFT within 2% of the best throughput"
            )
            + (f", p90 TTFT ≤ {options.max_ttft_p90:g} s" if options.max_ttft_p90 else ""),
            estimated_seconds=estimate_seconds(options, state.profile_schedule, DEFAULT_STARTUP_ESTIMATE),
            output_dir=str(output),
            replaced=tuple(replaced_options(serve_args)),
        )
        console.print(render_plan(plan))
        interactive = console.is_terminal and sys.stdin.isatty()
        if interactive and not options.yes:
            try:
                proceed = Confirm.ask("Proceed?", default=True, console=console)
            except (KeyboardInterrupt, EOFError):
                proceed = False
            if not proceed:
                console.print("Nothing was started.")
                return 1
    output.mkdir(parents=True, exist_ok=True)
    state.save(output)

    dashboard = Dashboard(model=info.model, gpus=info.gpus, workload=workload.description, output_dir=str(output))
    use_live = console.is_terminal and not options.no_tui
    reporter: LiveReporter | PlainReporter = (
        LiveReporter(dashboard, console) if use_live else PlainReporter(dashboard, console)
    )
    calibration = Calibration(options, serve_args, info, workload, sampler, state, output, dashboard, reporter)
    interrupted = False
    resume_hint = f"Progress is saved. Continue with:\n  sonar calibrate flashstates --resume {output}"
    with reporter:
        try:
            interrupted = asyncio.run(_supervise(calibration))
        except (ServerStartError, CalibrationError) as error:
            state.save(output)
            raise CalibrationError(f"{error}\n{resume_hint}") from None
        except Exception as error:
            state.save(output)
            log = output / "error.log"
            log.write_text(traceback.format_exc())
            raise CalibrationError(f"{type(error).__name__}: {error} (traceback in {log})\n{resume_hint}") from None
    if interrupted:
        state.save(output)
        console.print(f"[yellow]Stopped.[/] {resume_hint}", highlight=False)
        return 130
    if not use_live:
        console.print(render(dashboard))
    return 0


async def _supervise(calibration: Calibration) -> bool:
    """Run the calibration; the first Ctrl-C stops it cleanly, a second one
    exits at once. Returns whether it was interrupted."""
    loop = asyncio.get_running_loop()
    main = asyncio.create_task(calibration.run())
    presses = 0

    def on_interrupt() -> None:
        nonlocal presses
        presses += 1
        if presses == 1:
            calibration.reporter.event("Stopping: shutting down the server and saving progress (Ctrl-C again to force)")
            main.cancel()
            return
        server = calibration.server
        if server is not None and server._process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server._process.pid, signal.SIGKILL)
        calibration.state.save(calibration.output)
        os._exit(130)

    loop.add_signal_handler(signal.SIGINT, on_interrupt)
    try:
        await main
    except asyncio.CancelledError:
        if presses:
            return True
        raise
    finally:
        loop.remove_signal_handler(signal.SIGINT)
    return False
