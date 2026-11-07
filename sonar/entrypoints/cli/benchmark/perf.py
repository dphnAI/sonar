# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
import argparse

from aphrodite.benchmarks.perf import add_cli_args, main
from aphrodite.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from aphrodite.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkPerfSubcommand(BenchmarkSubcommandBase):
    """The `perf` subcommand for `aphrodite bench`."""

    name = "perf"
    help = "Benchmark single-request prefill and decode throughput."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
