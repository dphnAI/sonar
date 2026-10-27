# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`aphrodite tokenizer`: serve only a model's tokenizer."""

import argparse

import uvloop

from aphrodite.entrypoints.cli.types import CLISubcommand
from aphrodite.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from aphrodite.entrypoints.serve.utils.api_utils import APHRODITE_SUBCMD_PARSER_EPILOG
from aphrodite.utils.argparse_utils import FlexibleArgumentParser

DESCRIPTION = (
    "Start a tokenizer-only HTTP server. It loads the model's tokenizer and config but no "
    "weights or engine, and serves /tokenize and /detokenize with the same behavior as "
    "`aphrodite serve`."
)


class TokenizerSubcommand(CLISubcommand):
    """The `tokenizer` subcommand for the Aphrodite CLI."""

    name = "tokenizer"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from aphrodite.entrypoints.launchers.tokenizer.entry import run_tokenizer_server

        if getattr(args, "model_tag", None) is not None:
            args.model = args.model_tag
        uvloop.run(run_tokenizer_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Start a tokenizer-only server (no engine or GPU).",
            description=DESCRIPTION,
            usage="aphrodite tokenizer [model_tag] [options]",
        )
        parser = make_arg_parser(parser)
        parser.epilog = APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [TokenizerSubcommand()]
