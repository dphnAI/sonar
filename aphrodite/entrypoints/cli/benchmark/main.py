# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import logging
import os
import sys
import typing

from aphrodite import envs
from aphrodite.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from aphrodite.entrypoints.cli.types import CLISubcommand
from aphrodite.entrypoints.common.startup import APHRODITE_SUBCMD_PARSER_EPILOG
from aphrodite.logger import init_logger

if typing.TYPE_CHECKING:
    from aphrodite.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


def maybe_exec_rust_bench() -> None:
    if sys.argv[1:3] != ["bench", "serve"] or not envs.APHRODITE_USE_RUST_BENCH:
        return

    rust_cli = envs.APHRODITE_RUST_FRONTEND_PATH
    if rust_cli is None:
        raise RuntimeError(
            "APHRODITE_USE_RUST_BENCH=1 requires APHRODITE_RUST_FRONTEND_PATH to resolve to the aphrodite-rs binary."
        )

    logger.info("Delegating `aphrodite bench serve` to Rust binary at %s.", rust_cli)
    os.execv(rust_cli, [rust_cli, "bench", "serve", *sys.argv[3:]])


def _import_bench_subcommand_modules() -> None:
    # Imported lazily so `BenchmarkSubcommandBase` subclasses register only
    # when `aphrodite bench` is actually invoked.
    import aphrodite.entrypoints.cli.benchmark.forward  # noqa: F401
    import aphrodite.entrypoints.cli.benchmark.latency  # noqa: F401
    import aphrodite.entrypoints.cli.benchmark.mm_processor  # noqa: F401
    import aphrodite.entrypoints.cli.benchmark.perf  # noqa: F401
    import aphrodite.entrypoints.cli.benchmark.serve  # noqa: F401
    import aphrodite.entrypoints.cli.benchmark.startup  # noqa: F401
    import aphrodite.entrypoints.cli.benchmark.sweep  # noqa: F401
    import aphrodite.entrypoints.cli.benchmark.throughput  # noqa: F401


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the Aphrodite CLI."""

    name = "bench"
    help = "Aphrodite bench subcommand."

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
            usage=f"aphrodite {self.name} <bench_type> [options]",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        # Only build the nested bench subparsers when the user is actually
        # invoking `bench`; otherwise we'd drag in imports
        # unnecessarily on every `aphrodite --help` and `aphrodite serve`.
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
                    usage=f"aphrodite {self.name} {cmd_cls.name} [options]",
                )
                cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
                cmd_cls.add_cli_args(cmd_subparser)
                cmd_subparser.epilog = APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd=f"{self.name} {cmd_cls.name}")
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
