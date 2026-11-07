# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["build_nightly_commits.sh", "reconcile_platform_wheels.sh"])
@pytest.mark.parametrize("build_status", [0, 1])
@pytest.mark.parametrize("torch_helper_location", ["root", "tools", "untracked"])
def test_backlog_uses_current_helper_and_restores_checkout(tmp_path, script, build_status, torch_helper_location):
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github/scripts" / script).read_text()
    function = (
        "build_with_current_rust_helper() ("
        + source.split("build_with_current_rust_helper() (", 1)[1].split("\n)", 1)[0]
        + "\n)"
    )

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "tools").mkdir()
    helper = tmp_path / "tools/build_rust.py"
    helper.write_text("old helper\n")
    torch_helper = tmp_path / "tools/use_existing_torch.py"
    root_torch_helper = tmp_path / "use_existing_torch.py"
    original_torch_helper = torch_helper if torch_helper_location == "tools" else root_torch_helper
    original_torch_helper.write_text("old torch helper\n")
    original_torch_helper.chmod(0o640)
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "old")
    old = git("rev-parse", "HEAD")
    helper.write_text("new helper\n")
    root_torch_helper.unlink(missing_ok=True)
    torch_helper.write_text("new torch helper\n")
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "new")
    new = git("rev-parse", "HEAD")
    git("checkout", "--detach", old)
    original_torch_helper.chmod(0o640)
    if torch_helper_location == "untracked":
        torch_helper.write_text("untracked torch override\n")
        torch_helper.chmod(0o640)
    torch_assertion = (
        'test "$(<tools/use_existing_torch.py)" = "new torch helper"\n'
        if script == "reconcile_platform_wheels.sh"
        else ""
    )
    (tmp_path / "docker").mkdir()
    build = tmp_path / "docker/export_wheels.sh"
    build.write_text(
        "#!/bin/bash\nset -eu\n"
        'test "$(<tools/build_rust.py)" = "new helper"\n'
        'test "$SONAR_VERSION_OVERRIDE" = "0.24.0"\n' + torch_assertion + f"exit {build_status}\n"
    )
    build.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            function + '\ntarget_commit="$1"\n'
            "PLATFORM_BUILD_SCRIPT=./docker/export_wheels.sh\noutput_dir=wheels\n"
            'build_with_current_rust_helper "$2" 0.24.0',
            "test",
            new,
            old,
        ],
        cwd=tmp_path,
    )
    assert result.returncode == build_status
    assert helper.read_text() == "old helper\n"
    assert git("diff", "--name-only") == ""
    if torch_helper_location == "root":
        assert not torch_helper.exists()
    else:
        expected = "old torch helper\n" if torch_helper_location == "tools" else "untracked torch override\n"
        assert torch_helper.read_text() == expected
        assert torch_helper.stat().st_mode & 0o777 == 0o640
    if torch_helper_location != "tools":
        assert root_torch_helper.read_text() == "old torch helper\n"
