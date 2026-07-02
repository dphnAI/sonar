# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"


def run_helper(
    *args: str,
    env: dict[str, str] | None = None,
    path: str | None = None,
) -> list[str]:
    helper_env = {"PATH": path or os.environ["PATH"]}
    if env:
        helper_env.update(env)
    result = subprocess.run(
        ["bash", str(HELPER), *args],
        check=True,
        env=helper_env,
        stdout=subprocess.PIPE,
        text=True,
    )
    return shlex.split(result.stdout)


def option_values(args: list[str], option: str) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == option]


def build_args(args: list[str]) -> dict[str, str]:
    values = {}
    for value in option_values(args, "--build-arg"):
        key, arg_value = value.split("=", 1)
        values[key] = arg_value
    return values


def test_release_metadata_args_prefer_pipeline_id() -> None:
    args = run_helper(
        "cu130-ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "abc123",
            "BUILDKITE_PIPELINE_ID": "pipe-uuid",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/aphrodite/builds/1",
            "RELEASE_VERSION": "v0.20.0",
        },
    )

    assert build_args(args) == {
        "APHRODITE_BUILD_COMMIT": "abc123",
        "APHRODITE_BUILD_PIPELINE": "pipe-uuid",
        "APHRODITE_BUILD_URL": "https://buildkite.example/aphrodite/builds/1",
        "APHRODITE_IMAGE_TAG": "aphrodite/aphrodite-openai:v0.20.0-cu130-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/aphrodite-release-repo:"
        f"abc123-{os.uname().machine}-cu130-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_nightly_metadata_args_fall_back_to_pipeline_slug() -> None:
    args = run_helper(
        "ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "def456",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/aphrodite/builds/2",
            "NIGHTLY": "1",
        },
    )

    assert build_args(args) == {
        "APHRODITE_BUILD_COMMIT": "def456",
        "APHRODITE_BUILD_PIPELINE": "release",
        "APHRODITE_BUILD_URL": "https://buildkite.example/aphrodite/builds/2",
        "APHRODITE_IMAGE_TAG": "aphrodite/aphrodite-openai:nightly-def456-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/aphrodite-release-repo:"
        f"def456-{os.uname().machine}-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_local_metadata_args_use_local_overrides() -> None:
    args = run_helper(
        env={
            "APHRODITE_IMAGE_TAG": "local/test:dev",
            "APHRODITE_BUILD_COMMIT": "localsha",
            "APHRODITE_BUILD_PIPELINE": "local-pipeline",
            "APHRODITE_BUILD_URL": "https://buildkite.example/local",
        },
    )

    assert build_args(args) == {
        "APHRODITE_BUILD_COMMIT": "localsha",
        "APHRODITE_BUILD_PIPELINE": "local-pipeline",
        "APHRODITE_BUILD_URL": "https://buildkite.example/local",
        "APHRODITE_IMAGE_TAG": "local/test:dev",
    }
    assert option_values(args, "--tag") == ["local/test:dev"]


def test_release_version_lookup_failure_falls_back_to_commit(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    buildkite_agent = fake_bin / "buildkite-agent"
    buildkite_agent.write_text("#!/bin/sh\nexit 1\n")
    buildkite_agent.chmod(0o755)

    args = run_helper(
        "cu129",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "fallback123",
            "BUILDKITE_PIPELINE_SLUG": "release",
        },
        path=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert build_args(args)["APHRODITE_IMAGE_TAG"] == ("aphrodite/aphrodite-openai:vfallback123-cu129")


def test_vllm_openai_image_embeds_metadata_contract() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    for expected in (
        "ARG APHRODITE_BUILD_COMMIT",
        "ARG APHRODITE_BUILD_PIPELINE",
        "ARG APHRODITE_BUILD_URL",
        "ARG APHRODITE_IMAGE_TAG",
        "APHRODITE_BUILD_COMMIT=${APHRODITE_BUILD_COMMIT:-unknown}",
        "APHRODITE_BUILD_PIPELINE=${APHRODITE_BUILD_PIPELINE:-local}",
        "APHRODITE_BUILD_URL=${APHRODITE_BUILD_URL:-}",
        "APHRODITE_IMAGE_TAG=${APHRODITE_IMAGE_TAG:-local/aphrodite-openai:dev}",
        'ai.aphrodite.build.commit="${APHRODITE_BUILD_COMMIT}"',
        'ai.aphrodite.build.pipeline="${APHRODITE_BUILD_PIPELINE}"',
        'ai.aphrodite.build.url="${APHRODITE_BUILD_URL}"',
        'ai.aphrodite.image.tag="${APHRODITE_IMAGE_TAG}"',
    ):
        assert expected in dockerfile
