#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify native artifacts required by a CUDA wheel."""

from __future__ import annotations

import argparse
import stat
import sys
import zipfile
from pathlib import Path


def verify(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}

    missing: list[str] = []
    rust_binary = "sonar/sonar-rs"
    if rust_binary not in entries:
        missing.append(rust_binary)
    else:
        mode = entries[rust_binary].external_attr >> 16
        if not mode & stat.S_IXUSR:
            missing.append(f"{rust_binary} (not executable)")

    if not any(
        name.startswith("sonar/_rust_tool_parser") and name.endswith(".so") and "/" not in name.removeprefix("sonar/")
        for name in entries
    ):
        missing.append("sonar/_rust_tool_parser*.so")

    deepgemm_dir = "sonar/third_party/deep_gemm/"
    shim = f"{deepgemm_dir}_C.py"
    extension = f"{deepgemm_dir}_C_extension.abi3.so"
    if shim not in entries:
        missing.append(shim)
    bindings = [name for name in entries if name.startswith(f"{deepgemm_dir}_C") and name.endswith(".so")]
    if bindings != [extension]:
        missing.append(f"{extension} (exactly one stable-ABI binding; no legacy bindings)")

    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise RuntimeError(f"{wheel} is missing required native artifacts:\n{formatted}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify(args.wheel)
    print(f"Verified CUDA wheel native artifacts: {args.wheel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
