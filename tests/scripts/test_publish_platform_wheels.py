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
def test_publish_platform_wheels_filters_distribution(tmp_path: Path, channel: str, backend: str) -> None:
    root = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    local_version = "" if backend == "cuda" else f"+{backend}"
    current = f"sonar-1.0{local_version}-cp311-abi3-linux_x86_64.whl"
    stale = f"sonar-1.0{local_version}-cp310-abi3-linux_x86_64.whl"
    previous = f"sonar-0.9{local_version}-cp311-abi3-linux_x86_64.whl"
    legacy = f"sonar_engine-1.0{local_version}-cp311-abi3-linux_x86_64.whl"
    unrelated = "other_engine-1.0-cp311-abi3-linux_x86_64.whl"
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
            "x86_64",
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
    index_dir = output / "whl" / channel / backend / "x86_64"
    document = (index_dir / "simple/sonar/index.html").read_text()
    assert document == (index_dir / "index.html").read_text()
    assert current in document
    assert previous in document
    assert stale not in document
    assert legacy not in document
    assert unrelated not in document
    assert f"#sha256={hashlib.sha256(current.encode()).hexdigest()}" in document
