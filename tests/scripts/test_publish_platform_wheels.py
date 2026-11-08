# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("channel", ["release", "nightly"])
@pytest.mark.parametrize("backend", ["cuda", "cpu", "metal"])
@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_publish_platform_wheels_filters_distribution(
    tmp_path: Path, channel: str, backend: str, architecture: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    local_version = "" if backend == "cuda" else f"+{backend}"
    current = f"sonar-1.0{local_version}-cp311-abi3-linux_{architecture}.whl"
    stale = f"sonar-1.0{local_version}-cp310-abi3-linux_{architecture}.whl"
    previous = f"sonar-0.9{local_version}-cp311-abi3-linux_{architecture}.whl"
    legacy = f"sonar_engine-1.0{local_version}-cp311-abi3-linux_{architecture}.whl"
    unrelated = f"other_engine-1.0-cp311-abi3-linux_{architecture}.whl"
    for name in (current, legacy, unrelated):
        (wheel_dir / name).write_bytes(name.encode())

    calls = tmp_path / "calls.jsonl"
    remote_wheels = tmp_path / "remote.json"
    remote_wheels.write_text(json.dumps([current, stale, previous, legacy, unrelated]))
    rclone = tmp_path / "rclone"
    rclone.write_text(
        f"#!{sys.executable}\n"
        "import fnmatch, json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['RCLONE_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "state = Path(os.environ['REMOTE_WHEELS'])\n"
        "names = json.loads(state.read_text())\n"
        "if args[0] == 'lsf':\n"
        "    pattern = args[args.index('--include') + 1]\n"
        "    print('\\n'.join(name for name in names if fnmatch.fnmatchcase(name, pattern)))\n"
        "elif args[0] == 'deletefile':\n"
        "    names.remove(args[1].rsplit('/', 1)[-1])\n"
        "    state.write_text(json.dumps(names))\n"
        "elif args[0] != 'copyto':\n"
        "    raise SystemExit(f'Unexpected command: {args}')\n"
    )
    rclone.chmod(0o755)
    output = tmp_path / "indexes"
    subprocess.run(
        [
            "bash",
            str(root / ".github/scripts/publish_platform_wheels.sh"),
            str(wheel_dir),
            channel,
            backend,
            architecture,
            str(output),
        ],
        cwd=root,
        env={
            **os.environ,
            "B2_S3_BUCKET": "test-bucket",
            "B2_PUBLIC_BASE_URL": "https://wheels.example.test",
            "RCLONE_BIN": str(rclone),
            "RCLONE_CALLS": str(calls),
            "REMOTE_WHEELS": str(remote_wheels),
            "RUNNER_TEMP": str(tmp_path),
            "PRUNE_RELEASE_VERSION": "1.0",
        },
        check=True,
    )

    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    uploaded = [Path(command[1]).name for command in commands if command[0] == "copyto"]
    assert uploaded == [current, "index.html"]
    deleted = [command[1].rsplit("/", 1)[-1] for command in commands if command[0] == "deletefile"]
    assert deleted == [stale]
    index_dir = output / "whl" / channel / backend / architecture
    document = (index_dir / "simple/sonar/index.html").read_text()
    assert document == (index_dir / "index.html").read_text()
    assert current in document
    assert previous in document
    assert stale not in document
    assert legacy not in document
    assert unrelated not in document
    assert f"#sha256={hashlib.sha256(current.encode()).hexdigest()}" in document
    if backend == "cuda":
        assert "--torch-backend cu130" in document
    assert f"/platform-wheels/{channel}/{backend}/{architecture}/wheels/" in document


@pytest.mark.parametrize("backend", ["cuda", "cpu", "rocm"])
def test_reconciled_commit_index_uses_pinned_generator(tmp_path: Path, backend: str) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github/scripts/reconcile_platform_wheels.sh").read_text()
    functions = "\n".join(
        name + "() {" + source.split(name + "() {", 1)[1].split("\n}", 1)[0] + "\n}"
        for name in ("encode_path", "write_commit_index")
    )
    rclone = tmp_path / "rclone"
    rclone.write_text('#!/bin/bash\nset -eu\ncp "$2" "$INDEX_CAPTURE"\n')
    rclone.chmod(0o755)
    document_path = tmp_path / "index.html"
    wheel = f"sonar-0.25.1.dev1+{backend}-cp311-abi3-linux_aarch64.whl"
    # Historical checkouts need not contain the current index generator.
    subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            functions + '\npublication_tools="$1"\nrclone="$2"\n'
            'public_root="https://wheels.example.test/$PLATFORM_BACKEND/aarch64"\n'
            'remote_root="b2:test/$PLATFORM_BACKEND/aarch64"\n'
            'write_commit_index abc123 "$3" deadbeef\n',
            "test",
            str(root / ".github/scripts"),
            str(rclone),
            wheel,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "PLATFORM_BACKEND": backend,
            "PLATFORM_ARCHITECTURE": "aarch64",
            "INDEX_CAPTURE": str(document_path),
        },
        check=True,
    )
    document = document_path.read_text()
    assert ("--torch-backend cu130" in document) == (backend == "cuda")
    assert f"/{backend}/aarch64/commits/abc123" in document
    assert "#sha256=deadbeef" in document
