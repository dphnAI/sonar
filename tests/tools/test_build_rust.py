# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("version", [None, "0.24.0.dev1"])
def test_rust_build_without_git_metadata(tmp_path, version):
    """The cached Docker stage inherits pyproject.toml but has no .git."""
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "tools").mkdir()
    shutil.copy2(root / "tools/build_rust.py", tmp_path / "tools/build_rust.py")
    shutil.copy2(root / "pyproject.toml", tmp_path / "pyproject.toml")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SETUPTOOLS_SCM_PRETEND_VERSION") or key == "SONAR_RS_BUILD_VERSION":
            env.pop(key)
    if version:
        env["SONAR_RS_BUILD_VERSION"] = version
    # Command help runs the actual setuptools initialization, without Cargo.
    result = subprocess.run(
        [sys.executable, "tools/build_rust.py", "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "build_rust" in result.stdout
