# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import logging
import os
import sys
import typing

from sonar import envs
from sonar.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from sonar.entrypoints.cli.types import CLISubcommand
from sonar.entrypoints.common.startup import SONAR_SUBCMD_PARSER_EPILOG
from sonar.logger import init_logger

if typing.TYPE_CHECKING:
    from sonar.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


def maybe_exec_rust_bench() -> None:
    if sys.argv[1:3] != ["bench", "serve"] or not envs.SONAR_USE_RUST_BENCH:
        return

    rust_cli = envs.SONAR_RUST_FRONTEND_PATH
    if rust_cli is None:
        raise RuntimeError(
            "SONAR_USE_RUST_BENCH=1 requires SONAR_RUST_FRONTEND_PATH to resolve to the sonar-rs binary."
        )

    logger.info("Delegating `sonar bench serve` to Rust binary at %s.", rust_cli)
    os.execv(rust_cli, [rust_cli, "bench", "serve", *sys.argv[3:]])


def _import_bench_subcommand_modules() -> None:
    # Imported lazily so `BenchmarkSubcommandBase` subclasses register only
    # when `sonar bench` is actually invoked.
    import sonar.entrypoints.cli.benchmark.forward  # noqa: F401
    import sonar.entrypoints.cli.benchmark.latency  # noqa: F401
    import sonar.entrypoints.cli.benchmark.mm_processor  # noqa: F401
    import sonar.entrypoints.cli.benchmark.perf  # noqa: F401
    import sonar.entrypoints.cli.benchmark.serve  # noqa: F401
    import sonar.entrypoints.cli.benchmark.startup  # noqa: F401
    import sonar.entrypoints.cli.benchmark.sweep  # noqa: F401
    import sonar.entrypoints.cli.benchmark.throughput  # noqa: F401


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the Sonar CLI."""

    name = "bench"
    help = "Sonar bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"sonar {self.name} <bench_type> [options]",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        # Only build the nested bench subparsers when the user is actually
        # invoking `bench`; otherwise we'd drag in imports
        # unnecessarily on every `sonar --help` and `sonar serve`.
        # Scan for the first positional arg so global flags (e.g. `-v`)
        # before the subcommand don't break detection.
        first_positional = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
        if first_positional == self.name:
            previous_disable_level = logging.root.manager.disable
            logging.disable(logging.INFO)
            try:
                _import_bench_subcommand_modules()
            finally:
                logging.disable(previous_disable_level)
            for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
                cmd_subparser = bench_subparsers.add_parser(
                    cmd_cls.name,
                    help=cmd_cls.help,
                    description=cmd_cls.help,
                    usage=f"sonar {self.name} {cmd_cls.name} [options]",
                )
                cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
                cmd_cls.add_cli_args(cmd_subparser)
                cmd_subparser.epilog = SONAR_SUBCMD_PARSER_EPILOG.format(subcmd=f"{self.name} {cmd_cls.name}")
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
