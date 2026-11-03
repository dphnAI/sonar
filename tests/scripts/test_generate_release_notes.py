# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite Engine project

import os
import subprocess
import sys
from pathlib import Path


def git(repository: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args],
        env=env,
        text=True,
    ).strip()


def commit(repository: Path, subject: str, author: str, email: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": author,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": "Sync Bot",
        "GIT_COMMITTER_EMAIL": "sync@example.com",
    }
    subprocess.run(
        ["git", "-C", str(repository), "commit", "--allow-empty", "-m", subject],
        check=True,
        env=env,
    )
    return git(repository, "rev-parse", "HEAD")


def test_release_notes_list_rebased_commits(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)

    commit(repository, "Previous release", "Maintainer", "maintainer@example.com")
    subprocess.run(["git", "-C", str(repository), "tag", "v1.0.0"], check=True)
    first = commit(
        repository,
        "[sync] Add upstream feature (#12345)",
        "Upstream Author",
        "123+upstream-author@users.noreply.github.com",
    )
    second = commit(repository, "fix: retain fork behavior", "Fork Author", "fork@example.com")
    subprocess.run(["git", "-C", str(repository), "tag", "v1.1.0"], check=True)

    output = tmp_path / "notes.md"
    script = Path(__file__).parents[2] / ".github/scripts/generate_release_notes.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tag",
            "v1.1.0",
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
    )

    notes = output.read_text()
    assert f"[sync] Add upstream feature (#12345) ([`{first[:9]}`]" in notes
    assert "by @upstream-author" in notes
    assert f"fix: retain fork behavior ([`{second[:9]}`]" in notes
    assert "by Fork Author" in notes
    assert "compare/v1.0.0...v1.1.0" in notes
    assert "Previous release" not in notes
