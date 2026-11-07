# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from sonar.benchmarks.mm_processor import add_cli_args, main
from sonar.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from sonar.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkMMProcessorSubcommand(BenchmarkSubcommandBase):
    """The `mm-processor` subcommand for `sonar bench`."""

    name = "mm-processor"
    help = "Benchmark multimodal processor latency across different configurations."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
