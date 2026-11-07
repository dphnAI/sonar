# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from sonar.collect_env import main as collect_env_main
from sonar.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from sonar.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class CollectEnvSubcommand(CLISubcommand):
    """The `collect-env` subcommand for the Sonar CLI."""

    name = "collect-env"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Collect information about the environment."""
        collect_env_main()

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        return subparsers.add_parser(
            "collect-env",
            help="Start collecting environment information.",
            description="Start collecting environment information.",
            usage="sonar collect-env",
        )


def cmd_init() -> list[CLISubcommand]:
    return [CollectEnvSubcommand()]
