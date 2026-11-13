# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`sonar calibrate`: tune serving settings by measuring them."""

import argparse
import sys
import typing

from sonar.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from sonar.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

FLASHSTATES_DESCRIPTION = """Find the --prefill-cost-coefficients that serve a workload best.

First profiles uncached prefill time and fits the cost model, then replays a
multi-turn workload under cache pressure with candidate eviction ratios,
starting a fresh server for each. Put your serve command after '--':

  sonar calibrate flashstates -- MODEL --tensor-parallel-size 2 \\
      --enable-prefix-caching --mamba-frontier-checkpoint --mamba-prefill-export \\
      --prefix-cache-eviction gds --prefix-cache-leaf-last

Progress is saved; resume an interrupted run with --resume DIR.
"""


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Split `sonar calibrate ...` arguments at the first '--'. The part after
    it is the serve command, kept out of the CLI parser."""
    if argv[:1] != ["calibrate"] or "--" not in argv:
        return argv, None
    split = argv.index("--")
    return argv[:split], argv[split + 1 :]


class CalibrateSubcommand(CLISubcommand):
    """The `calibrate` subcommand for the Sonar CLI."""

    name = "calibrate"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if getattr(args, "calibrate_target", None) is None:
            args.calibrate_parser.print_help()
            sys.exit(2)
        from sonar.benchmarks.calibration.main import run

        sys.exit(run(args))

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Measure and tune serving settings for a model and workload.",
            description="Measure and tune serving settings for a model and workload.",
            usage="sonar calibrate <target> [options] -- MODEL [serve options]",
        )
        # Plain argparse help: this target has few options, so list them all.
        targets = parser.add_subparsers(dest="calibrate_target", parser_class=argparse.ArgumentParser)
        flashstates = targets.add_parser(
            "flashstates",
            help="Find the best --prefill-cost-coefficients for FlashStates eviction.",
            description=FLASHSTATES_DESCRIPTION,
            usage="sonar calibrate flashstates [options] -- MODEL [serve options]",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        from sonar.benchmarks.calibration.options import add_cli_args

        add_cli_args(flashstates)
        parser.set_defaults(calibrate_parser=parser)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [CalibrateSubcommand()]
