# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import importlib.metadata
import typing

from sonar.entrypoints.cli.types import CLISubcommand
from sonar.entrypoints.common.startup import SONAR_SUBCMD_PARSER_EPILOG
from sonar.logger import init_logger

if typing.TYPE_CHECKING:
    from sonar.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


class RunBatchSubcommand(CLISubcommand):
    """The `run-batch` subcommand for Sonar CLI."""

    name = "run-batch"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from sonar.entrypoints.launchers.run_batch import main as run_batch_main

        logger.info(
            "Sonar batch processing API version %s",
            importlib.metadata.version("sonar"),
        )
        logger.info("args: %s", args)

        # Start the Prometheus metrics server.
        # LLMEngine uses the Prometheus client
        # to publish metrics at the /metrics endpoint.
        if args.enable_metrics:
            from prometheus_client import start_http_server

            logger.info("Prometheus metrics enabled")
            # --url is deprecated in favour of --host, so prefer --host and fall
            # back to it only when --host was not given.
            start_http_server(port=args.port, addr=args.host or args.url)
        else:
            logger.info("Prometheus metrics disabled")

        asyncio.run(run_batch_main(args))

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        from sonar.entrypoints.launchers.run_batch import make_arg_parser

        run_batch_parser = subparsers.add_parser(
            self.name,
            help="Run batch prompts and write results to file.",
            description=(
                "Run batch prompts using Sonar's OpenAI-compatible API.\nSupports local or HTTP input/output files."
            ),
            usage="sonar run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model>",
        )
        run_batch_parser = make_arg_parser(run_batch_parser)
        run_batch_parser.epilog = SONAR_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return run_batch_parser


def cmd_init() -> list[CLISubcommand]:
    return [RunBatchSubcommand()]
