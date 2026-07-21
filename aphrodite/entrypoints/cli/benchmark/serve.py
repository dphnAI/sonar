# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os
import sys
from pathlib import Path

from aphrodite.benchmarks.serve import add_cli_args
from aphrodite.benchmarks.serve import main as python_main
from aphrodite.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from aphrodite.logger import init_logger
from aphrodite.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)
_RUST_CLI_PATH = Path(__file__).resolve().parents[3] / "aphrodite-rs"
_RUST_SUPPORTED_DATASETS = frozenset(
    {
        "custom",
        "hf",
        "prefix_repetition",
        "random",
        "random-mm",
        "random-rerank",
        "sharegpt",
        "sonnet",
        "speed_bench",
    }
)
_RUST_SUPPORTED_BACKENDS = frozenset(
    {
        "aphrodite",
        "aphrodite-pooling",
        "aphrodite-rerank",
        "openai",
        "openai-chat",
        "openai-embeddings",
        "openai-embeddings-chat",
        "vllm",
        "vllm-pooling",
        "vllm-rerank",
    }
)


def _rust_unsupported_reason(args: argparse.Namespace) -> str | None:
    if args.dataset_name not in _RUST_SUPPORTED_DATASETS:
        return f"dataset {args.dataset_name!r} is not supported by the Rust benchmark"
    if args.backend not in _RUST_SUPPORTED_BACKENDS:
        return f"backend {args.backend!r} is not supported by the Rust benchmark"
    return None


def _maybe_exec_rust_bench(args: argparse.Namespace) -> None:
    if reason := _rust_unsupported_reason(args):
        logger.info("Using Python benchmark: %s.", reason)
        return

    if not _RUST_CLI_PATH.is_file():
        logger.warning(
            "Rust benchmark binary not found at %s; falling back to Python.",
            _RUST_CLI_PATH,
        )
        return

    rust_cli = str(_RUST_CLI_PATH)
    logger.info("Delegating `aphrodite bench serve` to Rust binary at %s.", rust_cli)
    os.execv(rust_cli, [rust_cli, "bench", "serve", *sys.argv[3:]])


class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """The `serve` subcommand for `aphrodite bench`."""

    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        _maybe_exec_rust_bench(args)
        python_main(args)
