# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of Aphrodite

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys

from aphrodite.logger import init_logger

logger = init_logger(__name__)


def main():
    import aphrodite.entrypoints.cli.benchmark.main
    import aphrodite.entrypoints.cli.collect_env
    import aphrodite.entrypoints.cli.launch
    import aphrodite.entrypoints.cli.openai
    import aphrodite.entrypoints.cli.run
    import aphrodite.entrypoints.cli.run_batch
    import aphrodite.entrypoints.cli.serve
    from aphrodite.entrypoints.serve.utils.api_utils import (
        APHRODITE_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )
    from aphrodite.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        aphrodite.entrypoints.cli.openai,
        aphrodite.entrypoints.cli.serve,
        aphrodite.entrypoints.cli.run,
        aphrodite.entrypoints.cli.launch,
        aphrodite.entrypoints.cli.benchmark.main,
        aphrodite.entrypoints.cli.collect_env,
        aphrodite.entrypoints.cli.run_batch,
    ]

    cli_env_setup()

    # For 'aphrodite bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from aphrodite import platforms

        if platforms.current_platform.is_unspecified():
            from aphrodite.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = FlexibleArgumentParser(
        description="Aphrodite CLI",
        epilog=APHRODITE_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("aphrodite-engine"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
