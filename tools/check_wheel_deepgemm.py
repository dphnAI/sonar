#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check installed DeepGEMM's stable-ABI layout and import its Python binding.

Run with each supported interpreter after installing the same wheel into that
interpreter's environment. One abi3 extension serves all supported Python minors.
"""

import importlib
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    package_name = "sonar.third_party.deep_gemm"
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        print(f"{package_name} is not installed", file=sys.stderr)
        return 1
    package_dir = Path(spec.origin).parent
    shim = package_dir / "_C.py"
    extensions = sorted(package_dir.glob("_C*.so"))
    expected = package_dir / "_C_extension.abi3.so"
    missing = []
    if not shim.is_file():
        missing.append("_C.py")
    if extensions != [expected]:
        missing.append("exactly one _C_extension.abi3.so and no legacy bindings")
    print(f"DeepGEMM stable binding: shim={shim.is_file()}, extensions={[path.name for path in extensions]}")
    if missing:
        return 1

    binding = importlib.import_module(f"{package_name}._C")
    print(f"Imported DeepGEMM for Python {sys.version_info.major}.{sys.version_info.minor}: {binding.__file__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
