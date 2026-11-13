# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Command-line options of `sonar calibrate flashstates`. Kept free of heavy
imports because every `sonar` invocation builds this parser."""

import argparse
import dataclasses
from dataclasses import dataclass

CODING_AGENTS = "coding-agents"
# Warmup seconds, measurement seconds and trials per effort level.
EFFORTS = {"quick": (300.0, 600.0, 4), "standard": (600.0, 1200.0, 6), "thorough": (900.0, 1800.0, 8)}


@dataclass(slots=True)
class CalibrationOptions:
    workload: str = CODING_AGENTS
    sessions: int | None = None
    trials: int | None = None
    effort: str = "standard"
    warmup: float | None = None
    duration: float | None = None
    objective: str = "throughput"
    max_ttft_p90: float | None = None
    profile_max_tokens: int | None = None
    output_dir: str | None = None
    resume: str | None = None
    seed: int = 0
    startup_timeout: float = 3600.0
    yes: bool = False
    no_tui: bool = False

    @property
    def schedule(self) -> tuple[float, float, int]:
        warmup, duration, trials = EFFORTS[self.effort]
        return (
            self.warmup if self.warmup is not None else warmup,
            self.duration if self.duration is not None else duration,
            self.trials if self.trials is not None else trials,
        )


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    workload = parser.add_argument_group("workload")
    workload.add_argument(
        "--workload",
        default=CODING_AGENTS,
        help="Sessions to replay: 'coding-agents' (public coding-agent traces, downloaded once), "
        "'synthetic[:key=value,...]', or a Weka-format trace file or directory of your own traffic. "
        "Default: %(default)s.",
    )
    workload.add_argument(
        "--sessions",
        type=int,
        help="Concurrent sessions per trial. Default: enough for 1.5x the KV cache capacity.",
    )
    workload.add_argument("--seed", type=int, default=0, help="Seed for session order and prompts. Default: 0.")

    search = parser.add_argument_group("search")
    search.add_argument(
        "--effort",
        choices=sorted(EFFORTS),
        default="standard",
        help="Per-trial warmup and measurement, and the trial count: quick 5+10 min x4, "
        "standard 10+20 min x6, thorough 15+30 min x8. Default: %(default)s.",
    )
    search.add_argument("--trials", type=int, help="Number of search trials (overrides --effort).")
    search.add_argument("--warmup", type=float, help="Seconds of cache warmup per trial (overrides --effort).")
    search.add_argument("--duration", type=float, help="Seconds of measurement per trial (overrides --effort).")
    search.add_argument(
        "--objective",
        choices=("throughput", "latency"),
        default="throughput",
        help="'throughput': most output tokens per second, ties to the lower p90 TTFT. 'latency': lowest p90 "
        "TTFT among ratios within 2%% of the best throughput. Default: %(default)s.",
    )
    search.add_argument(
        "--max-ttft-p90", type=float, metavar="SECONDS", help="p90 TTFT target; slower ratios are penalised."
    )
    search.add_argument(
        "--profile-max-tokens",
        type=int,
        help="Longest prompt to profile. Default: the workload's 95th percentile prompt, at least 16384.",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", help="Where to write results. Default: ./sonar-calibration/<model>-<time>.")
    run.add_argument("--resume", metavar="DIR", help="Continue an interrupted calibration from its output directory.")
    run.add_argument(
        "--startup-timeout",
        type=float,
        default=3600.0,
        metavar="SECONDS",
        help="How long each server may take to become healthy. Default: %(default)s.",
    )
    run.add_argument("-y", "--yes", action="store_true", help="Start without confirming the plan.")
    run.add_argument("--no-tui", action="store_true", help="Print plain progress lines instead of the live view.")


def options_from_args(args: argparse.Namespace) -> CalibrationOptions:
    fields = {field.name for field in dataclasses.fields(CalibrationOptions)}
    return CalibrationOptions(**{name: getattr(args, name) for name in fields if hasattr(args, name)})
