# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import uvloop

from sonar.entrypoints.cli.types import CLISubcommand
from sonar.entrypoints.common.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from sonar.entrypoints.common.startup import SONAR_SUBCMD_PARSER_EPILOG
from sonar.entrypoints.launchers.render.entry import run_launch_fastapi
from sonar.logger import init_logger
from sonar.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

DESCRIPTION = "Launch individual Sonar components."


class LaunchSubcommandBase(CLISubcommand):
    """The base class of subcommands for `sonar launch`."""

    help: str

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        """Add the CLI arguments to the parser.

        By default, adds the standard Sonar serving arguments.
        Subclasses can override to add component-specific arguments.
        """
        make_arg_parser(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError


class RenderSubcommand(LaunchSubcommandBase):
    """The `render` subcommand for `sonar launch`."""

    name = "render"
    help = "Launch a GPU-less rendering server (preprocessing and postprocessing only)."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        uvloop.run(run_launch_fastapi(args))


class LaunchSubcommand(CLISubcommand):
    """The `launch` subcommand for the Sonar CLI.

    Uses nested sub-subcommands so each component can define its own
    arguments independently (e.g. ``sonar launch render``).
    """

    name = "launch"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        args.launch_command(args)

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        launch_parser = subparsers.add_parser(
            self.name,
            help=DESCRIPTION,
            description=DESCRIPTION,
            usage=f"sonar {self.name} <component> [options]",
        )
        launch_subparsers = launch_parser.add_subparsers(required=True, dest="launch_component")

        for cmd_cls in LaunchSubcommandBase.__subclasses__():
            cmd_subparser = launch_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"sonar {self.name} {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(launch_command=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
            cmd_subparser.epilog = SONAR_SUBCMD_PARSER_EPILOG.format(subcmd=f"{self.name} {cmd_cls.name}")

        return launch_parser


def cmd_init() -> list[CLISubcommand]:
    return [LaunchSubcommand()]
