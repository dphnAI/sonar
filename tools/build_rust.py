# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setuptools-rust build entry for Rust artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension
from setuptools_scm import get_version

ROOT_DIR = Path(__file__).resolve().parents[1]
SONAR_RS_BUILD_VERSION = "SONAR_RS_BUILD_VERSION"

# Select Sonar release tags, excluding crate tags such as "proto-v0.3.0".
SONAR_GIT_DESCRIBE_COMMAND = [
    "git",
    "describe",
    "--dirty",
    "--tags",
    "--long",
    "--abbrev=40",
    "--match",
    "v[0-9]*",
]


def prepare_build_environment() -> str | None:
    """Set the device-independent Sonar source version for Rust artifacts."""
    version = os.getenv(SONAR_RS_BUILD_VERSION) or None
    if version is None:
        try:
            version = get_version(
                root=ROOT_DIR,
                git_describe_command=SONAR_GIT_DESCRIBE_COMMAND,
            )
        except LookupError:
            return None

    os.environ[SONAR_RS_BUILD_VERSION] = version
    return version


def rust_extensions(*, optional: bool = False) -> list[RustExtension]:
    return [
        RustExtension(
            target="sonar.sonar-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "sonar-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
        ),
        RustExtension(
            target="sonar._rust_tool_parser",
            path="rust/src/parser/python/Cargo.toml",
            features=["pyo3/abi3-py38"],
            binding=Binding.PyO3,
            optional=optional,
            py_limited_api=True,
        ),
    ]


def rust_py_extension_module_names() -> list[str]:
    module_names = []
    for extension in rust_extensions():
        if extension.binding != Binding.PyO3:
            continue

        for target_name in extension.target.values():
            if target_name.startswith("sonar._rust_"):
                module_names.append(target_name.rsplit(".", 1)[-1])

    return module_names


def build_binary(build_rust_args: list[str]) -> None:
    os.chdir(ROOT_DIR)
    (ROOT_DIR / "sonar").mkdir(exist_ok=True)
    setup(
        name="sonar-rust-frontend-build",
        # Docker's cached Rust stage has pyproject.toml but no Git metadata.
        # Give this build-only distribution a version so setuptools-scm does
        # not try to infer one again during setup().
        version=os.getenv(SONAR_RS_BUILD_VERSION) or "0.0.0",
        packages=[],
        rust_extensions=rust_extensions(optional=False),
        script_args=["build_rust", "--quiet", "--inplace", *build_rust_args],
    )


def main() -> None:
    prepare_build_environment()
    build_binary(sys.argv[1:])


if __name__ == "__main__":
    main()
