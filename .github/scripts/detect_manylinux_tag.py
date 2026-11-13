#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Detect and apply the manylinux tag supported by a wheel's binaries.

This changes only the wheel filename. CUDA libraries remain external, so
`auditwheel repair` cannot be used to graft dependencies into the wheel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auditwheel.error import (
    AuditwheelError,
    NonPlatformWheelError,
    WheelToolsError,
)
from auditwheel.wheel_abi import analyze_wheel_abi
from auditwheel.wheeltools import get_wheel_architecture, get_wheel_libc


def detect_platform_tag(wheel: Path) -> str:
    """Return the most precise manylinux tag supported by the wheel."""
    try:
        architecture = get_wheel_architecture(wheel.name)
    except (WheelToolsError, NonPlatformWheelError):
        architecture = None

    try:
        libc = get_wheel_libc(wheel.name)
    except WheelToolsError:
        libc = None

    wheel_info = analyze_wheel_abi(
        libc,
        architecture,
        wheel,
        frozenset(),
        disable_isa_ext_check=False,
        allow_graft=False,
    )
    return wheel_info.sym_policy.name


def rename_wheel(wheel: Path, platform_tag: str) -> Path:
    """Replace the platform component of a wheel filename."""
    parts = wheel.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unrecognized wheel filename: {wheel.name}")
    parts[-1] = platform_tag
    destination = wheel.with_name("-".join(parts) + ".whl")
    if destination != wheel:
        wheel.rename(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    if not args.wheel.is_file():
        parser.error(f"wheel does not exist: {args.wheel}")

    try:
        platform_tag = detect_platform_tag(args.wheel)
        destination = rename_wheel(args.wheel, platform_tag)
    except (AuditwheelError, OSError, ValueError) as error:
        print(
            f"Failed to retag {args.wheel.name}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Detected platform tag: {platform_tag}", file=sys.stderr)
    if destination != args.wheel:
        print(
            f"Renamed {args.wheel.name} to {destination.name}",
            file=sys.stderr,
        )
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
