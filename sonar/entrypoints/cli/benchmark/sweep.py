# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from sonar.benchmarks.sweep.cli import add_cli_args, main
from sonar.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from sonar.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkSweepSubcommand(BenchmarkSubcommandBase):
    """The `sweep` subcommand for `sonar bench`."""

    name = "sweep"
    help = "Benchmark for a parameter sweep."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
