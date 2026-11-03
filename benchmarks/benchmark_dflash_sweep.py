#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import regex as re
import requests
from transformers import AutoTokenizer

REQUEST_LOG_RE = re.compile(
    r"Request completed - E2E time: (?P<e2e>[0-9.]+)s"
    r"(?:, TTFT: (?P<ttft>[0-9.]+)s)?"
    r"(?:, Prefill: (?P<prefill_tokens>\d+) tokens "
    r"\((?P<prefill_tps>[0-9.]+) tokens/s\))?"
    r"(?:, Decode: (?P<decode_tokens>\d+) tokens "
    r"\((?P<decode_tps>[0-9.]+) tokens/s\))?"
)


def parse_sizes(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def iter_output(proc: subprocess.Popen[str], queue: Queue[str], log_path: Path) -> None:
    assert proc.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_file:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            queue.put(line.rstrip())


def drain_request_logs(queue: Queue[str]) -> list[dict[str, float | int]]:
    parsed: list[dict[str, float | int]] = []
    while True:
        try:
            line = queue.get_nowait()
        except Empty:
            return parsed
        match = REQUEST_LOG_RE.search(line)
        if match is None:
            continue
        item: dict[str, float | int] = {}
        for key, value in match.groupdict().items():
            if value is None:
                continue
            item[key] = int(value) if key.endswith("tokens") else float(value)
        parsed.append(item)


def wait_for_server(base_url: str, proc: subprocess.Popen[str], timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s: {last_error}")


def make_prompt(tokenizer: Any, token_count: int) -> str:
    seed = (
        "In a quiet observatory above the city, researchers compared model "
        "outputs, cache behavior, speculative decoding traces, and throughput "
        "measurements while documenting every result carefully. "
    )
    ids: list[int] = []
    while len(ids) < token_count + 8:
        ids = tokenizer.encode(seed * max(1, (token_count // 24) + 2), add_special_tokens=False)
        seed += seed
    return tokenizer.decode(ids[:token_count], skip_special_tokens=True)


def existing_cases(csv_path: Path) -> set[tuple[int, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {
            (int(row["prefill_tokens_requested"]), int(row["decode_tokens_requested"]))
            for row in csv.DictReader(f)
            if row.get("status") == "ok"
        }


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    fields = [
        "prefill_tokens_requested",
        "decode_tokens_requested",
        "status",
        "client_e2e_s",
        "client_decode_tps",
        "server_e2e_s",
        "server_ttft_s",
        "server_prefill_tokens",
        "server_prefill_tps",
        "server_decode_tokens",
        "server_decode_tps",
        "completion_tokens",
        "prompt_tokens",
        "error",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_benchmark(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dflash_sweep.csv"
    log_path = out_dir / "server.log"
    config_path = out_dir / "config.json"

    prefill_sizes = parse_sizes(args.prefill_sizes)
    decode_sizes = parse_sizes(args.decode_sizes)
    completed = existing_cases(csv_path) if args.resume else set()

    tokenizer_model = args.tokenizer_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=args.trust_remote_code)
    prompts = {size: make_prompt(tokenizer, size) for size in prefill_sizes}

    spec_config = None
    if not args.no_dflash:
        spec_config = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    server_cmd = [
        args.aphrodite_bin,
        "run",
        args.model,
        "--tokenizer",
        tokenizer_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "-gmu",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-model-len",
        str(args.max_model_len),
    ]
    if spec_config is not None:
        server_cmd.extend(["--speculative-config", json.dumps(spec_config)])
    if args.language_model_only:
        server_cmd.append("--language-model-only")
    if args.extra_server_arg:
        for extra in args.extra_server_arg:
            server_cmd.extend(extra.split())

    config_path.write_text(
        json.dumps(
            {
                "server_cmd": server_cmd,
                "prefill_sizes": prefill_sizes,
                "decode_sizes": decode_sizes,
                "speculative_config": spec_config,
                "tokenizer_model": tokenizer_model,
                "args": vars(args),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("APHRODITE_NO_USAGE_STATS", "1")
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    queue: Queue[str] = Queue()
    Thread(target=iter_output, args=(proc, queue, log_path), daemon=True).start()
    base_url = f"http://127.0.0.1:{args.port}"

    try:
        wait_for_server(base_url, proc, args.startup_timeout_s)
        if args.warmup_tokens > 0:
            drain_request_logs(queue)
            warmup_prompt = prompts[prefill_sizes[0]]
            warmup_payload = {
                "model": args.model,
                "prompt": warmup_prompt,
                "max_tokens": args.warmup_tokens,
                "temperature": args.temperature,
                "ignore_eos": True,
            }
            print(
                f"warmup prefill={prefill_sizes[0]} decode={args.warmup_tokens}",
                flush=True,
            )
            response = requests.post(
                f"{base_url}/v1/completions",
                json=warmup_payload,
                timeout=args.request_timeout_s,
            )
            response.raise_for_status()
            time.sleep(args.log_wait_s)
            drain_request_logs(queue)

        for prefill in prefill_sizes:
            prompt = prompts[prefill]
            for decode in decode_sizes:
                case = (prefill, decode)
                if case in completed:
                    print(f"skip prefill={prefill} decode={decode}", flush=True)
                    continue

                drain_request_logs(queue)
                payload = {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": decode,
                    "temperature": args.temperature,
                    "ignore_eos": True,
                }
                if args.min_p is not None:
                    payload["min_p"] = args.min_p
                print(f"run prefill={prefill} decode={decode}", flush=True)
                start = time.perf_counter()
                row: dict[str, Any] = {
                    "prefill_tokens_requested": prefill,
                    "decode_tokens_requested": decode,
                }
                try:
                    response = requests.post(
                        f"{base_url}/v1/completions",
                        json=payload,
                        timeout=args.request_timeout_s,
                    )
                    client_e2e = time.perf_counter() - start
                    row["client_e2e_s"] = round(client_e2e, 6)
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage") or {}
                    completion_tokens = usage.get("completion_tokens", decode)
                    prompt_tokens = usage.get("prompt_tokens", prefill)
                    row["completion_tokens"] = completion_tokens
                    row["prompt_tokens"] = prompt_tokens
                    row["client_decode_tps"] = round(completion_tokens / client_e2e, 4)
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = repr(exc)

                deadline = time.monotonic() + args.log_wait_s
                latest_log: dict[str, Any] | None = None
                while time.monotonic() < deadline:
                    logs = drain_request_logs(queue)
                    if logs:
                        latest_log = logs[-1]
                        break
                    time.sleep(0.2)
                if latest_log:
                    row.update(
                        {
                            "server_e2e_s": latest_log.get("e2e"),
                            "server_ttft_s": latest_log.get("ttft"),
                            "server_prefill_tokens": latest_log.get("prefill_tokens"),
                            "server_prefill_tps": latest_log.get("prefill_tps"),
                            "server_decode_tokens": latest_log.get("decode_tokens"),
                            "server_decode_tps": latest_log.get("decode_tps"),
                        }
                    )
                append_row(csv_path, row)
                print(json.dumps(row), flush=True)
    finally:
        terminate_process(proc)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3.5-4B-DFlash")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--aphrodite-bin", default="aphrodite")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2242)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--no-dflash", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--prefill-sizes", default="256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--decode-sizes", default="256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--output-dir", default="benchmarks/results/dflash_qwen35_4b")
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-timeout-s", type=int, default=1800)
    parser.add_argument("--request-timeout-s", type=int, default=7200)
    parser.add_argument("--log-wait-s", type=float, default=10)
    parser.add_argument("--warmup-tokens", type=int, default=512)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--extra-server-arg", action="append")
    args = parser.parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
