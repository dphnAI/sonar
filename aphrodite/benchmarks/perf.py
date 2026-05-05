# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Single-request model performance benchmark.

This intentionally mirrors the compact output style of exllamav3's
``eval/perf.py`` while running through Aphrodite's normal offline engine.
"""

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from types import TracebackType

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from aphrodite.inputs import TokensPrompt
from aphrodite.outputs import RequestOutput

ESC = "\u001b"
COL_DEFAULT = f"{ESC}[0m"
COL_GREEN = f"{ESC}[32;1m"
COL_YELLOW = f"{ESC}[33;1m"


@contextmanager
def _suppress_startup_logs(enabled: bool):
    if not enabled:
        yield
        return

    previous_disable_level = logging.root.manager.disable
    previous_env_level = os.environ.get("APHRODITE_LOGGING_LEVEL")
    aphrodite_logger = logging.getLogger("aphrodite")
    previous_aphrodite_level = aphrodite_logger.level
    previous_handler_levels = [handler.level for handler in aphrodite_logger.handlers]
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    os.environ["APHRODITE_LOGGING_LEVEL"] = "ERROR"
    aphrodite_logger.setLevel(logging.ERROR)
    for handler in aphrodite_logger.handlers:
        handler.setLevel(logging.ERROR)
    logging.disable(logging.INFO)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)
        logging.disable(previous_disable_level)
        if previous_env_level is None:
            os.environ.pop("APHRODITE_LOGGING_LEVEL", None)
        else:
            os.environ["APHRODITE_LOGGING_LEVEL"] = previous_env_level
        aphrodite_logger.setLevel(previous_aphrodite_level)
        for handler, level in zip(aphrodite_logger.handlers, previous_handler_levels):
            handler.setLevel(level)


class _StartupProgress:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.progress: Progress | None = None
        self.console_file = None

    def __enter__(self) -> "_StartupProgress":
        if self.enabled:
            self.console_file = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=Console(file=self.console_file),
                transient=True,
            )
            self.progress.start()
            self.progress.add_task("Loading, compiling, and capturing model...", total=None)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.progress is not None:
            if exc_type is None:
                for task_id in self.progress.task_ids:
                    self.progress.update(task_id, description="Model ready")
            self.progress.stop()
        if self.console_file is not None:
            self.console_file.close()


def _get_lengths(max_length: int) -> list[int]:
    length = 256
    lengths = [length]
    while length < max_length:
        length = min(length * 2, max_length)
        lengths.append(length)
    return lengths


def _get_vocab_size(llm) -> int:  # type: ignore[no-untyped-def]
    vocab_size = getattr(llm.model_config.hf_config, "vocab_size", None)
    tokenizer = llm.get_tokenizer()
    try:
        tokenizer_size = len(tokenizer)
    except TypeError:
        tokenizer_size = len(tokenizer.get_vocab())

    if vocab_size is None:
        return tokenizer_size
    return min(vocab_size, tokenizer_size)


def _make_prompt(
    rng: np.random.Generator,
    length: int,
    vocab_size: int,
) -> TokensPrompt:
    token_ids = rng.integers(0, vocab_size, size=length, dtype=np.int64)
    return TokensPrompt(prompt_token_ids=token_ids.tolist())


def _prefill_time(output: RequestOutput, wall_time: float) -> float:
    metrics = output.metrics
    if metrics is None:
        return wall_time

    prefill_time = metrics.first_token_ts - metrics.scheduled_ts
    if prefill_time <= 0:
        return wall_time
    return prefill_time


def _decode_time(output: RequestOutput, wall_time: float) -> float:
    metrics = output.metrics
    if metrics is None:
        return wall_time

    decode_time = metrics.last_token_ts - metrics.first_token_ts
    if decode_time <= 0:
        return wall_time
    return decode_time


def _run_generate(llm, prompt: TokensPrompt, max_tokens: int):  # type: ignore[no-untyped-def]
    from aphrodite import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=max_tokens,
        detokenize=False,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
    end = time.perf_counter()
    return outputs[0], end - start


def _measure_prefill(
    args: argparse.Namespace,
    llm,
    lengths: Sequence[int],
    rng: np.random.Generator,
    vocab_size: int,
    *,
    warmup: bool = False,
) -> dict[int, float]:
    results: dict[int, float] = {}
    for length in lengths:
        prompt = _make_prompt(rng, length, vocab_size)
        output, wall_time = _run_generate(llm, prompt, max_tokens=1)
        tokens_per_second = length / _prefill_time(output, wall_time)
        results[length] = tokens_per_second
        if not warmup:
            print(f"Length  {length:6}: {COL_GREEN}{tokens_per_second:10.2f}{COL_DEFAULT} tokens/s", flush=True)
    return results


def _measure_generate(
    args: argparse.Namespace,
    llm,
    contexts: Sequence[int],
    rng: np.random.Generator,
    vocab_size: int,
    *,
    warmup: bool = False,
) -> dict[int, float]:
    results: dict[int, float] = {}
    for context_len in contexts:
        # Aphrodite needs at least one prompt token. Label the first case as
        # context 0 to match exllamav3's eval/perf.py output.
        prompt_len = max(context_len, 1)
        prompt = _make_prompt(rng, prompt_len, vocab_size)

        # The first generated token is produced by the prefill step. Request
        # one extra token so the measured decode interval covers args.gen_tokens
        # decode steps.
        output, wall_time = _run_generate(llm, prompt, max_tokens=args.gen_tokens + 1)
        tokens_per_second = args.gen_tokens / _decode_time(output, wall_time)
        results[context_len] = tokens_per_second
        if not warmup:
            print(f"Context {context_len:6}: {COL_GREEN}{tokens_per_second:10.2f}{COL_DEFAULT} tokens/s", flush=True)
    return results


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model_tag",
        nargs="?",
        help="Model name or path. Equivalent to --model for this benchmark.",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        dest="model_dir",
        help="Model name or path, matching exllamav3 eval/perf.py.",
    )
    parser.add_argument(
        "-max_length",
        "--max-length",
        type=int,
        default=32768,
        help="Max context length to measure.",
    )
    parser.add_argument(
        "-chunk_size",
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size used for the default max-num-batched-tokens.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=100,
        help="Number of decode tokens to measure for each context length.",
    )
    parser.add_argument(
        "-spf",
        "--skip-prefill",
        action="store_true",
        help="Skip measuring prefill speed.",
    )
    parser.add_argument(
        "-swu",
        "--skip-warmup",
        action="store_true",
        help="Skip warmup passes.",
    )
    parser.add_argument(
        "--show-startup-logs",
        action="store_true",
        help="Show normal Aphrodite engine startup logs instead of the compact startup progress line.",
    )
    # Building engine CLI args can initialize parts of the platform layer.
    # Keep that quiet so `bench perf` has a compact exllamav3-like surface.
    with _suppress_startup_logs(enabled=True):
        from aphrodite.engine.arg_utils import EngineArgs

        EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False, disable_log_stats=False)


def main(args: argparse.Namespace) -> None:
    model = args.model_dir or args.model_tag or args.model
    if model is None:
        raise ValueError("aphrodite bench perf requires a model via MODEL, -m, or --model.")
    args.model = model

    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = args.chunk_size
    max_required_len = args.max_length + args.gen_tokens + 1
    if args.max_model_len is None:
        args.max_model_len = max_required_len

    # Keep request-level timing metrics enabled, but avoid interleaving the
    # normal "Request completed" log line with the perf.py-style table.
    logging.getLogger("aphrodite.v1.metrics.loggers").setLevel(logging.WARNING)

    hide_startup_logs = not args.show_startup_logs
    with _StartupProgress(enabled=hide_startup_logs):
        with _suppress_startup_logs(enabled=hide_startup_logs):
            from aphrodite import LLM
            from aphrodite.engine.arg_utils import EngineArgs

            engine_args = EngineArgs.from_cli_args(args)
            llm = LLM.from_engine_args(engine_args)

        assert llm.llm_engine.model_config.max_model_len >= max_required_len, (
            f"Please ensure max_model_len is at least {max_required_len} tokens for this benchmark."
        )

        vocab_size = _get_vocab_size(llm)
        rng = np.random.default_rng(args.seed)

        print(f" -- Model: {model}", flush=True)
        print(f" -- Chunk size: {args.chunk_size}", flush=True)
        print(flush=True)

        prefill_lengths = _get_lengths(args.max_length)
        generate_contexts = [0] + _get_lengths(max(args.max_length - 256, 256))

        if not args.skip_prefill:
            if not args.skip_warmup:
                warmup_prefill_lengths = _get_lengths(min(args.chunk_size, args.max_length))
                _measure_prefill(args, llm, warmup_prefill_lengths, rng, vocab_size, warmup=True)

            print(f"{COL_YELLOW}Prefill:{COL_DEFAULT}", flush=True)
            _measure_prefill(args, llm, prefill_lengths, rng, vocab_size)
            print(flush=True)

        if not args.skip_warmup:
            warmup_generate_contexts = [0] + _get_lengths(min(args.chunk_size, args.max_length))
            _measure_generate(args, llm, warmup_generate_contexts, rng, vocab_size, warmup=True)

        print(f"{COL_YELLOW}Generation{COL_DEFAULT}", flush=True)
        _measure_generate(args, llm, generate_contexts, rng, vocab_size)
        print(flush=True)
        del llm
