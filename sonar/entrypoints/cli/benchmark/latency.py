# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from aphrodite.benchmarks.latency import add_cli_args, main
from aphrodite.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from aphrodite.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkLatencySubcommand(BenchmarkSubcommandBase):
    """The `latency` subcommand for `aphrodite bench`."""

    name = "latency"
    help = "Benchmark the latency of a single batch of requests."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
