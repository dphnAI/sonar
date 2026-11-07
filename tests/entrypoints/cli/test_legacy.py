# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("entrypoint", "argument", "exit_code", "warns"),
    [
        ("main", "--version", 0, False),
        ("deprecated_main", "--version", 0, True),
        ("deprecated_main", "--unknown-legacy-option", 2, True),
    ],
)
def test_cli_migration_warning(tmp_path: Path, entrypoint: str, argument: str, exit_code: int, warns: bool) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-W",
            "ignore",
            "-c",
            f"from sonar.entrypoints.cli.main import {entrypoint}; {entrypoint}()",
            argument,
        ],
        cwd=tmp_path,
        env={**os.environ, "SONAR_PLUGINS": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    migration_warnings = [line for line in result.stderr.splitlines() if "aphrodite" in line.lower()]
    if warns:
        assert len(migration_warnings) == 1
        assert "sonar" in migration_warnings[0]
        assert "deprecated" in migration_warnings[0]
    else:
        assert not migration_warnings
    if argument == "--version":
        assert result.stdout.strip() == importlib.metadata.version("sonar")


def test_legacy_import_reports_migration(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "try:\n"
                "    import aphrodite\n"
                "except ImportError as error:\n"
                "    assert type(error) is ImportError, repr(error)\n"
                "    assert error.name == 'aphrodite'\n"
                "    assert 'sonar' in str(error)\n"
                "else:\n"
                "    raise AssertionError('Legacy import must not silently expose the Sonar API')\n"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
