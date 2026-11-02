# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import subprocess
import sys
from dataclasses import dataclass

from aphrodite import LLM, SamplingParams


@dataclass(frozen=True)
class ModelCase:
    name: str
    model: str
    prompt: str


DEFAULT_CASES = (
    ModelCase(
        "dense",
        "/home/alpindale/models/Qwen3-0.6B-exl3-4.0bpw",
        "Write one sentence about mountain lakes.",
    ),
    ModelCase(
        "moe",
        "/home/alpindale/models/Trinity-Nano-Preview-exl3-4.0bpw",
        "Write a short paragraph about mountain lakes.",
    ),
)


def _parse_case(value: str) -> ModelCase:
    parts = value.split("=", 2)
    if len(parts) == 1:
        return ModelCase(parts[0], parts[0], "Write one sentence about mountain lakes.")
    if len(parts) == 2:
        return ModelCase(parts[0], parts[1], "Write one sentence about mountain lakes.")
    return ModelCase(parts[0], parts[1], parts[2])


def _format_case(case: ModelCase) -> str:
    return f"{case.name}={case.model}={case.prompt}"


def _check_tokens(name: str, token_ids: list[int], text: str, min_tokens: int) -> None:
    if len(token_ids) < min_tokens:
        raise AssertionError(f"{name}: generated only {len(token_ids)} tokens")
    if not text.strip():
        raise AssertionError(f"{name}: generated empty text")

    most_common = max(token_ids.count(token_id) for token_id in set(token_ids))
    if most_common / len(token_ids) > 0.75:
        raise AssertionError(
            f"{name}: generated degenerate token stream; most common token appears {most_common}/{len(token_ids)} times"
        )
    if len(set(token_ids)) < min(8, len(token_ids) // 2):
        raise AssertionError(f"{name}: generated too few unique tokens: {len(set(token_ids))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXL3 generation sanity check for optimization work.")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Case as name=model=prompt. May be passed multiple times.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("-gmu", "--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    cases = tuple(_parse_case(case) for case in args.case) if args.case else DEFAULT_CASES
    if len(cases) > 1:
        for case in cases:
            cmd = [
                sys.executable,
                __file__,
                "--case",
                _format_case(case),
                "--max-tokens",
                str(args.max_tokens),
                "--min-tokens",
                str(args.min_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--dtype",
                args.dtype,
            ]
            subprocess.run(cmd, check=True)
        return

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7,
        top_p=0.95,
        ignore_eos=True,
    )

    for case in cases:
        llm = LLM(
            model=case.model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_prefix_caching=False,
        )
        outputs = llm.generate([case.prompt], sampling_params, use_tqdm=False)
        output = outputs[0].outputs[0]
        token_ids = list(output.token_ids)
        _check_tokens(case.name, token_ids, output.text, args.min_tokens)
        print(f"{case.name}: PASS tokens={len(token_ids)} unique={len(set(token_ids))} text={output.text[:80]!r}")
        del llm


if __name__ == "__main__":
    main()
