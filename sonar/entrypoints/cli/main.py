# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of Sonar.

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys

from sonar.logger import configure_logging_from_args, init_logger

logger = init_logger(__name__)


def main():
    # Keep the CPU verifier out of public help and avoid loading serving modules.
    if sys.argv[1:2] == ["verify"]:
        from sonar.proof.cli import main as verify_main

        verify_main(sys.argv[2:])
        return

    import sonar.entrypoints.cli.benchmark.main
    import sonar.entrypoints.cli.calibrate
    import sonar.entrypoints.cli.collect_env
    import sonar.entrypoints.cli.download_kernels
    import sonar.entrypoints.cli.launch
    import sonar.entrypoints.cli.openai
    import sonar.entrypoints.cli.preload
    import sonar.entrypoints.cli.run
    import sonar.entrypoints.cli.run_batch
    import sonar.entrypoints.cli.serve
    import sonar.entrypoints.cli.snapshot
    import sonar.entrypoints.cli.tokenizer
    from sonar.entrypoints.common.startup import SONAR_SUBCMD_PARSER_EPILOG, cli_env_setup
    from sonar.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        sonar.entrypoints.cli.calibrate,
        sonar.entrypoints.cli.openai,
        sonar.entrypoints.cli.serve,
        sonar.entrypoints.cli.run,
        sonar.entrypoints.cli.launch,
        sonar.entrypoints.cli.benchmark.main,
        sonar.entrypoints.cli.collect_env,
        sonar.entrypoints.cli.download_kernels,
        sonar.entrypoints.cli.preload,
        sonar.entrypoints.cli.run_batch,
        sonar.entrypoints.cli.snapshot,
        sonar.entrypoints.cli.tokenizer,
    ]

    if sys.argv[1:2] != ["snapshot"]:
        cli_env_setup()

    sonar.entrypoints.cli.benchmark.main.maybe_exec_rust_bench()

    # For 'sonar bench *' and the GPU-less 'sonar tokenizer': use CPU instead of
    # UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] in ("bench", "tokenizer"):
        logger.debug(
            "%s command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error",
            sys.argv[1],
        )
        from sonar import platforms

        if platforms.current_platform.is_unspecified():
            from sonar.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info("Unspecified platform detected, switching to CPU Platform instead.")

    parser = FlexibleArgumentParser(
        description="Sonar CLI",
        epilog=SONAR_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("sonar"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        if cmd_module is sonar.entrypoints.cli.snapshot:
            new_cmds = cmd_module.cmd_init(create_requested=sys.argv[1:3] == ["snapshot", "create"])
        else:
            new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    argv, serve_args = sonar.entrypoints.cli.calibrate.split_passthrough(sys.argv[1:])
    args = parser.parse_args(argv)
    if serve_args is not None:
        args.serve_args = serve_args
    if args.subparser in cmds:
        cmd = cmds[args.subparser]
        configure_logging_from_args(args)
        cmd.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


def deprecated_main():
    print(
        "WARNING: The 'aphrodite' CLI will be deprecated soon. Please use 'sonar' instead.",
        file=sys.stderr,
        flush=True,
    )
    return main()


if __name__ == "__main__":
    main()
