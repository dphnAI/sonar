# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Run mypy on changed files.

This script is designed to be used as a pre-commit hook. It runs mypy
on files that have been changed. It groups files into different mypy calls
based on their directory to avoid import following issues.

Usage:
    python tools/pre_commit/mypy.py <ci> <python_version> <changed_files...>

Args:
    ci: "1" if running in CI, "0" otherwise.
    python_version: Python version to use (e.g., "3.10") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.
"""

import subprocess
import sys

import regex as re

# After fixing errors resulting from changing follow_imports
# from "skip" to "silent", remove its directory from SEPARATE_GROUPS.
SEPARATE_GROUPS = [
    "tests",
    # v0 related
    "aphrodite/lora",
]

# TODO(woosuk): Include the code from Megatron and HuggingFace.
EXCLUDE = [
    "aphrodite/third_party",
    "aphrodite/vllm_flash_attn",
    "aphrodite/benchmarks",
    r"aphrodite/model_executor/models/[aA]",
    r"aphrodite/model_executor/models/[bB]",
    r"aphrodite/model_executor/models/[cC]",
    r"aphrodite/model_executor/models/[dD]",
    r"aphrodite/model_executor/models/[eE]",
    r"aphrodite/model_executor/models/[fF]",
    r"aphrodite/model_executor/models/[gG]",
    r"aphrodite/model_executor/models/[hH]",
    r"aphrodite/model_executor/models/[iI]",
    r"aphrodite/model_executor/models/[jJ]",
    r"aphrodite/model_executor/models/[kK]",
    r"aphrodite/model_executor/models/[lL]",
    r"aphrodite/model_executor/models/[mM]",
    r"aphrodite/model_executor/models/[nN]",
    r"aphrodite/model_executor/models/[oO]",
    r"aphrodite/model_executor/models/[pP]",
    r"aphrodite/model_executor/models/[qQ]",
    r"aphrodite/model_executor/models/[rR]",
    r"aphrodite/model_executor/models/[sS]",
    r"aphrodite/model_executor/models/[tT]",
    r"aphrodite/model_executor/models/[uU]",
    r"aphrodite/model_executor/models/[vV]",
    r"aphrodite/model_executor/models/[wW]",
    r"aphrodite/model_executor/models/[zZ]",
]


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """
    Group changed files into different mypy calls.

    Args:
        changed_files: List of changed files.

    Returns:
        A dictionary mapping file group names to lists of changed files.
    """
    exclude_pattern = re.compile(f"^{'|'.join(EXCLUDE)}.*")
    file_groups = {"": []}
    file_groups.update({k: [] for k in SEPARATE_GROUPS})
    for changed_file in changed_files:
        # Skip files which should be ignored completely
        if exclude_pattern.match(changed_file):
            continue
        # Group files by mypy call
        for directory in SEPARATE_GROUPS:
            if re.match(f"^{directory}.*", changed_file):
                file_groups[directory].append(changed_file)
                break
        else:
            if changed_file.startswith("aphrodite/"):
                file_groups[""].append(changed_file)
    return file_groups


def mypy(
    targets: list[str],
    python_version: str | None,
    follow_imports: str | None,
    file_group: str,
) -> int:
    """
    Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.10") or None to use
            the default mypy version.
        follow_imports: Value for the --follow-imports option or None to use
            the default mypy behavior.
        file_group: The file group name for logging purposes.

    Returns:
        The return code from mypy.
    """
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    print(f"$ {' '.join(args)} {file_group}")
    return subprocess.run(args + targets, check=False).returncode


def main():
    # sys.argv[1] is retained for CLI compatibility with CI/local mode.
    python_version = sys.argv[2]
    file_groups = group_files(sys.argv[3:])

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        follow_imports = "skip"
        if changed_files:
            returncode |= mypy(changed_files, python_version, follow_imports, file_group)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
