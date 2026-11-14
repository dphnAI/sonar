# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Terminal presentation of a calibration run.

The orchestrator updates a `Dashboard`; this module only renders it, either
as a live view or as plain timestamped lines."""

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from sonar.benchmarks.calibration.client import RunnerSnapshot
from sonar.benchmarks.calibration.fit import CostFit
from sonar.benchmarks.calibration.pressure import PressureStep
from sonar.benchmarks.calibration.profile import ProfileMeasurement
from sonar.benchmarks.calibration.search import SearchConfig, Selection, TrialMeasurement, format_ratio, select

STAGES = (("profile", "Profile"), ("pressure", "Probe"), ("search", "Search"), ("report", "Report"))
_BAR_WIDTH = 16
_PARTIAL_BLOCKS = " ▏▎▍▌▋▊▉"


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def format_coefficients(coefficients: tuple[float, float]) -> str:
    return f"--prefill-cost-coefficients {coefficients[0]:.4g} {coefficients[1]:.4g}"


@dataclass(slots=True)
class ServerView:
    label: str
    phase: str
    started: float
    last_line: str = ""


@dataclass(slots=True)
class TrialView:
    index: int
    total: int
    ratio: float
    sessions: int
    warmup: float
    duration: float
    phase: str = "starting"
    run_started: float | None = None
    snapshot: Callable[[], RunnerSnapshot] | None = None
    prefix_hit_rate: float | None = None
    kv_usage: float | None = None
    title: str = ""
    # Metrics when the run began, for a live hit rate before the window opens.
    baseline_metrics: dict[str, float] | None = None


@dataclass(slots=True)
class Dashboard:
    model: str
    gpus: int
    workload: str
    output_dir: str
    started: float = field(default_factory=time.monotonic)
    stage: str = "profile"
    server: ServerView | None = None
    profile_total: int = 0
    profile: list[ProfileMeasurement] = field(default_factory=list)
    profile_status: str = ""
    fit: CostFit | None = None
    trial: TrialView | None = None
    trials: list[TrialMeasurement] = field(default_factory=list)
    search_config: SearchConfig | None = None
    selection: Selection | None = None
    recommended: tuple[float, float] | None = None
    eta_seconds: float | None = None
    events: deque = field(default_factory=lambda: deque(maxlen=4))
    files: list[str] = field(default_factory=list)
    pressure: list[PressureStep] = field(default_factory=list)
    pressure_counts: list[int] = field(default_factory=list)
    sessions_note: str | None = None


# Building blocks


def _stepper(dashboard: Dashboard) -> Text:
    current = [name for name, _ in STAGES].index(dashboard.stage)
    text = Text()
    for index, (name, label) in enumerate(STAGES):
        if index:
            text.append("  ›  ", style="dim")
        if index < current or (dashboard.stage == "report" and dashboard.selection is not None):
            text.append(f"✓ {label}", style="green")
        elif index == current:
            suffix = ""
            if name in ("pressure", "search") and dashboard.trial is not None:
                kind = "probe" if name == "pressure" else "trial"
                suffix = f" ({kind} {dashboard.trial.index}/{dashboard.trial.total})"
            text.append(f"● {label}{suffix}", style="bold cyan")
        else:
            text.append(f"○ {label}", style="dim")
    return text


def _header(dashboard: Dashboard) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="right", no_wrap=True)
    gpus = f"{dashboard.gpus} GPU" + ("s" if dashboard.gpus != 1 else "")
    grid.add_row(
        Text.assemble((dashboard.model, "bold"), ("  ·  ", "dim"), gpus),
        Text.assemble(
            ("elapsed ", "dim"),
            format_duration(time.monotonic() - dashboard.started),
            ("   ETA ", "dim"),
            format_duration(dashboard.eta_seconds),
        ),
    )
    grid.add_row(Text(dashboard.workload, style="dim"), "")
    grid.add_row(_stepper(dashboard), "")
    return Panel(grid, title="[bold]Sonar · FlashStates calibration[/]", border_style="cyan")


def _server_panel(server: ServerView) -> Panel:
    elapsed = format_duration(time.monotonic() - server.started)
    body = Group(
        Spinner("dots", text=Text.assemble((server.phase, "bold"), (f"   {elapsed}", "dim"))),
        Text(server.last_line, style="dim", overflow="ellipsis", no_wrap=True),
    )
    return Panel(body, title=server.label, border_style="blue")


def _fit_line(fit: CostFit) -> Text:
    low, high = fit.ratio_interval
    if math.isnan(high):
        interval = ""
    else:
        interval = f" (90% {low:.3g}–{high:.3g})" if math.isfinite(high) else " (90% unbounded)"
    return Text.assemble(
        ("D ", "dim"),
        f"{fit.per_token:.4g} s/token",
        ("   E ", "dim"),
        f"{fit.per_token_per_100k:.4g}",
        ("   R = E/D ", "dim"),
        (f"{format_ratio(fit.ratio)}", "bold"),
        (interval, "dim"),
        ("   R² ", "dim"),
        f"{fit.r_squared:.4f}",
        ("   worst error ", "dim"),
        f"{fit.max_relative_error * 100:.1f}%",
    )


def _profile_panel(dashboard: Dashboard) -> Panel:
    done = len(dashboard.profile)
    total = max(dashboard.profile_total, 1)
    progress = Table.grid(padding=(0, 2))
    progress.add_row(
        ProgressBar(total=total, completed=done, width=40),
        Text(f"{done}/{dashboard.profile_total} requests"),
        Text(dashboard.profile_status, style="dim"),
    )
    by_length: dict[int, list[float]] = {}
    for measurement in dashboard.profile:
        by_length.setdefault(measurement.tokens, []).append(measurement.seconds)
    table = Table(box=None, pad_edge=False, header_style="bold dim")
    table.add_column("Prompt tokens", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Median prefill", justify="right")
    table.add_column("Prefill tok/s", justify="right")
    table.add_column("Model", justify="right")
    for length in sorted(by_length):
        values = sorted(by_length[length])
        median = values[len(values) // 2]
        model = f"{dashboard.fit.predict(length):.3f} s" if dashboard.fit else ""
        table.add_row(f"{length:,}", str(len(values)), f"{median:.3f} s", f"{length / median:,.0f}", model)
    parts: list[RenderableType] = [progress, Text(), table]
    if dashboard.fit is not None:
        parts += [Text(), _fit_line(dashboard.fit)]
    return Panel(Group(*parts), title="Prefill cost", border_style="magenta")


def _trial_bar(trial: TrialView) -> Text:
    width = 50
    total = trial.warmup + trial.duration
    elapsed = 0.0 if trial.run_started is None else min(total, time.monotonic() - trial.run_started)
    warmup_cells = round(width * trial.warmup / total)
    filled = round(width * elapsed / total)
    bar = Text()
    for cell in range(width):
        warm = cell < warmup_cells
        if cell < filled:
            bar.append("━", style="yellow" if warm else "cyan")
        else:
            bar.append("━", style="grey30")
    if trial.run_started is None:
        label = trial.phase
    elif elapsed < trial.warmup:
        label = f"warming the cache, {format_duration(trial.warmup - elapsed)} left"
    else:
        label = f"measuring, {format_duration(total - elapsed)} left"
    bar.append(f"  {label}", style="bold")
    return bar


def _trial_panel(dashboard: Dashboard, trial: TrialView) -> Panel:
    grid = Table.grid(padding=(0, 4))
    for _ in range(4):
        grid.add_column()
    snapshot = trial.snapshot() if trial.snapshot is not None else None
    per_gpu = (snapshot.output_tokens_per_second / dashboard.gpus) if snapshot else None

    def metric(label: str, value: str) -> Text:
        return Text.assemble((f"{label} ", "dim"), (value, "bold"))

    grid.add_row(
        metric("sessions", f"{snapshot.active_lanes}/{trial.sessions}" if snapshot else f"0/{trial.sessions}"),
        metric("in flight", str(snapshot.in_flight) if snapshot else "-"),
        metric("output", f"{per_gpu:,.0f} tok/s/GPU" if per_gpu is not None else "-"),
        metric("p90 TTFT", f"{snapshot.ttft_p90_recent:.2f} s" if snapshot and snapshot.ttft_p90_recent else "-"),
    )
    grid.add_row(
        metric("requests", f"{snapshot.completed:,} ok" if snapshot else "-"),
        metric("failed", str(snapshot.failed) if snapshot else "-"),
        metric("prefix hits", f"{trial.prefix_hit_rate * 100:.1f}%" if trial.prefix_hit_rate is not None else "-"),
        metric("KV usage", f"{trial.kv_usage * 100:.0f}%" if trial.kv_usage is not None else "-"),
    )
    title = trial.title or f"Trial {trial.index}/{trial.total} · R = {format_ratio(trial.ratio)}"
    return Panel(Group(_trial_bar(trial), Text(), grid), title=title, border_style="yellow")


def _bar(fraction: float) -> str:
    cells = max(0.0, min(1.0, fraction)) * _BAR_WIDTH
    whole = int(cells)
    partial = _PARTIAL_BLOCKS[int((cells - whole) * 8)]
    return ("█" * whole + (partial if whole < _BAR_WIDTH else "")).ljust(_BAR_WIDTH)


def candidates_table(
    trials: Sequence[TrialMeasurement],
    config: SearchConfig,
    gpus: int,
    *,
    running: float | None = None,
    selection: Selection | None = None,
) -> Table:
    selection = selection or (select(config, trials) if trials else None)
    hit_rates: dict[float, list[float]] = {}
    for trial in trials:
        if trial.prefix_hit_rate is not None:
            hit_rates.setdefault(trial.ratio, []).append(trial.prefix_hit_rate)
    table = Table(box=None, pad_edge=False, header_style="bold dim", expand=False)
    table.add_column("R = E/D", justify="right", no_wrap=True)
    table.add_column("Runs", justify="right")
    table.add_column("tok/s/GPU", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p90 TTFT", justify="right", no_wrap=True)
    table.add_column("Hits", justify="right")
    table.add_column("", no_wrap=True)
    candidates = selection.candidates if selection else ()
    best = max((candidate.output_tokens_per_second for candidate in candidates), default=0.0)
    for candidate in sorted(candidates, key=lambda c: c.ratio):
        chosen = selection is not None and candidate.ratio == selection.ratio
        label = format_ratio(candidate.ratio)
        if abs(candidate.ratio - config.fitted_ratio) <= 1e-12 * max(1.0, config.fitted_ratio):
            label += " fit"
        low, high = candidate.ci95
        ci = f"±{(high - low) / 2 / gpus:,.0f}" if math.isfinite(high) else "-"
        hits = hit_rates.get(candidate.ratio)
        table.add_row(
            Text(("★ " if chosen else "") + label, style="bold green" if chosen else ""),
            str(candidate.trials),
            f"{candidate.output_tokens_per_second / gpus:,.0f}",
            ci,
            f"{candidate.ttft_p90:.2f} s",
            f"{sum(hits) / len(hits) * 100:.1f}%" if hits else "-",
            Text(_bar(candidate.output_tokens_per_second / best if best else 0), style="green" if chosen else "cyan"),
        )
    if running is not None and not any(c.ratio == running for c in candidates):
        table.add_row(Text(format_ratio(running), style="yellow"), "", Text("running", style="yellow"))
    return table


def _candidates_panel(dashboard: Dashboard) -> Panel | None:
    if dashboard.search_config is None or (not dashboard.trials and dashboard.trial is None):
        return None
    running = dashboard.trial.ratio if dashboard.trial is not None else None
    table = candidates_table(
        dashboard.trials, dashboard.search_config, dashboard.gpus, running=running, selection=dashboard.selection
    )
    return Panel(table, title="Candidates", border_style="green")


def _result_panel(dashboard: Dashboard) -> Panel:
    selection = dashboard.selection
    assert selection is not None and dashboard.recommended is not None
    lines: list[RenderableType] = [
        Text("Recommended", style="bold"),
        Text.assemble(
            ("  " + format_coefficients(dashboard.recommended), "bold green"),
            (f"    (R = {format_ratio(selection.ratio)})", "dim"),
        ),
        Text(),
        Text(selection.reason),
    ]
    if selection.note:
        lines.append(Text(selection.note, style="yellow"))
    if dashboard.files:
        lines.append(Text())
        lines += [Text.assemble(("Wrote  ", "dim"), path) for path in dashboard.files]
    return Panel(Group(*lines), title="Result", border_style="green")


def _pressure_panel(dashboard: Dashboard) -> Panel:
    table = Table(box=None, pad_edge=False, header_style="bold dim")
    table.add_column("Sessions", justify="right")
    table.add_column("Prefix hits", justify="right")
    table.add_column("tok/s/GPU", justify="right")
    table.add_column("p90 TTFT", justify="right", no_wrap=True)
    table.add_column("", no_wrap=True)
    rates = [step.prefix_hit_rate or 0.0 for step in dashboard.pressure]
    best = max(rates, default=0.0)
    for step in dashboard.pressure:
        rate = step.prefix_hit_rate or 0.0
        table.add_row(
            str(step.sessions),
            f"{rate * 100:.1f}%",
            f"{step.output_tokens_per_second / dashboard.gpus:,.0f}",
            f"{step.ttft_p90:.2f} s",
            Text(_bar(rate / best if best else 0.0), style="cyan"),
        )
    trial = dashboard.trial
    if trial is not None:
        table.add_row(Text(str(trial.sessions), style="yellow"), Text("running", style="yellow"))
    for count in dashboard.pressure_counts:
        if count not in {step.sessions for step in dashboard.pressure} and (trial is None or trial.sessions != count):
            table.add_row(Text(str(count), style="dim"), Text("pending", style="dim"))
    parts: list[RenderableType] = [
        Text(
            "Finding the largest session count whose prefix hits stay high: idle sessions must compete "
            "for the cache, or eviction order cannot matter.",
            style="dim",
        ),
        Text(),
        table,
    ]
    if dashboard.sessions_note:
        parts += [Text(), Text(dashboard.sessions_note, style="bold")]
    return Panel(Group(*parts), title="Cache pressure", border_style="blue")


def render(dashboard: Dashboard) -> RenderableType:
    parts: list[RenderableType] = [_header(dashboard)]
    if dashboard.stage == "report" and dashboard.selection is not None:
        parts.append(_result_panel(dashboard))
    elif dashboard.server is not None:
        parts.append(_server_panel(dashboard.server))
    elif dashboard.stage == "profile":
        parts.append(_profile_panel(dashboard))
    elif dashboard.trial is not None:
        parts.append(_trial_panel(dashboard, dashboard.trial))
    if dashboard.stage == "pressure":
        parts.append(_pressure_panel(dashboard))
    elif dashboard.stage == "search" and dashboard.fit is not None:
        setup: list[RenderableType] = [_fit_line(dashboard.fit)]
        if dashboard.sessions_note:
            setup.append(Text(dashboard.sessions_note, style="dim"))
        parts.append(Panel(Group(*setup), title="Setup", border_style="magenta"))
    if (candidates := _candidates_panel(dashboard)) is not None:
        parts.append(candidates)
    for event in dashboard.events:
        parts.append(Text(event, style="dim", no_wrap=True, overflow="ellipsis"))
    parts.append(
        Text.assemble(("Output ", "dim"), dashboard.output_dir, ("   ·   Ctrl-C stops and can be resumed", "dim"))
    )
    return Group(*parts)


# Plan


@dataclass(frozen=True, slots=True)
class Plan:
    model: str
    gpus: int
    workload: str
    workload_details: str
    profile_requests: int
    profile_min_tokens: int
    profile_max_tokens: int
    trials: int
    warmup: float
    duration: float
    sessions: str
    objective: str
    estimated_seconds: float
    output_dir: str
    replaced: tuple[str, ...]


def render_plan(plan: Plan) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("Model", f"{plan.model} on {plan.gpus} GPU" + ("s" if plan.gpus != 1 else ""))
    table.add_row("Workload", plan.workload)
    table.add_row("", Text(plan.workload_details, style="dim"))
    table.add_row(
        "1. Profile",
        f"{plan.profile_requests} uncached prefills from {plan.profile_min_tokens:,} to "
        f"{plan.profile_max_tokens:,} tokens",
    )
    table.add_row(
        "2. Search",
        f"{plan.trials} trials, each a fresh server, {format_duration(plan.warmup)} warmup and "
        f"{format_duration(plan.duration)} measurement",
    )
    table.add_row("Sessions", plan.sessions)
    table.add_row("Objective", plan.objective)
    table.add_row("Estimated time", Text(format_duration(plan.estimated_seconds), style="bold"))
    table.add_row("Output", plan.output_dir)
    managed = "--host, --port and --prefill-cost-coefficients are set by the calibration"
    if plan.replaced:
        managed += f" (replacing your {', '.join(plan.replaced)})"
    table.add_row("Note", Text(managed, style="dim"))
    return Panel(table, title="[bold]Calibration plan[/]", border_style="cyan")


# Output modes


class LiveReporter:
    def __init__(self, dashboard: Dashboard, console: Console) -> None:
        self.dashboard = dashboard
        self.console = console
        self._live = Live(
            get_renderable=lambda: render(dashboard), console=console, refresh_per_second=4, transient=False
        )

    def __enter__(self) -> "LiveReporter":
        self._live.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._live.refresh()
        self._live.__exit__(*exc_info)

    def event(self, message: str) -> None:
        self.dashboard.events.append(f"{time.strftime('%H:%M:%S')}  {message}")

    def tick(self) -> None:
        pass


class PlainReporter:
    """Timestamped lines for logs and non-interactive terminals."""

    def __init__(self, dashboard: Dashboard, console: Console, interval: float = 30.0) -> None:
        self.dashboard = dashboard
        self.console = console
        self.interval = interval
        self._last_status = time.monotonic()

    def __enter__(self) -> "PlainReporter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def event(self, message: str) -> None:
        self.console.print(f"{time.strftime('%H:%M:%S')}  {message}", highlight=False, soft_wrap=True)

    def tick(self) -> None:
        if time.monotonic() - self._last_status < self.interval:
            return
        self._last_status = time.monotonic()
        status = status_line(self.dashboard)
        if status:
            self.event(status)


def status_line(dashboard: Dashboard) -> str:
    """One line describing the current activity."""
    if dashboard.server is not None:
        server = dashboard.server
        return f"{server.label}: {server.phase} ({format_duration(time.monotonic() - server.started)})"
    if dashboard.stage == "profile":
        return f"Profiling: {len(dashboard.profile)}/{dashboard.profile_total} requests. {dashboard.profile_status}"
    trial = dashboard.trial
    if trial is None or trial.snapshot is None or trial.run_started is None:
        return ""
    snapshot = trial.snapshot()
    elapsed = time.monotonic() - trial.run_started
    phase = "warmup" if elapsed < trial.warmup else "measuring"
    p90 = f"{snapshot.ttft_p90_recent:.2f} s" if snapshot.ttft_p90_recent else "-"
    return (
        f"{trial.title or f'Trial {trial.index}/{trial.total}'} {phase} "
        f"{format_duration(elapsed)}/{format_duration(trial.warmup + trial.duration)}: "
        f"{snapshot.output_tokens_per_second / dashboard.gpus:,.0f} output tok/s/GPU, p90 TTFT {p90}, "
        f"{snapshot.completed:,} requests, {snapshot.failed} failed"
    )
