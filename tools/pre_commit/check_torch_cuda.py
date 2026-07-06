# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import regex as re

_TORCH_CUDA_PATTERNS = [
    r"\btorch\.cuda\.(empty_cache|synchronize|device_count|current_device|memory_reserved|memory_allocated|max_memory_allocated|max_memory_reserved|reset_peak_memory_stats|memory_stats|set_device|device\()\b",
    r"\btorch\.cuda\.(manual_seed|manual_seed_all)\b",
    r"\bwith\storch\.cuda\.device\b",
    r"\bcuda_device_count_stateless\(\)\b",
]

ALLOWED_FILES = {
    "aphrodite/platforms/",
    "aphrodite/device_allocator/",
    "aphrodite/vllm_flash_attn/",
    "benchmarks/",
    "examples/",
    "tests/",
    "aphrodite/model_executor/layers/quantization/exl3.py",
    "aphrodite/distributed/weight_transfer/ipc_engine.py",
}


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for pattern in _TORCH_CUDA_PATTERNS:
        for match in re.finditer(pattern, content, re.MULTILINE):
            line_num = content[: match.start() + 1].count("\n") + 1
            matched_text = match.group(0)
            if "manual_seed" in matched_text:
                print(f"{path}:{line_num}: Found {matched_text} API call. Use set_random_seed instead.")
                return 1
            print(f"{path}:{line_num}: Found torch.cuda API call. Use torch.accelerator where possible instead.")
            return 1
    return 0


def main() -> int:
    returncode = 0
    for filename in sys.argv[1:]:
        if any(filename.startswith(prefix) for prefix in ALLOWED_FILES):
            continue
        returncode |= scan_file(filename)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
