# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the offline example scripts, one test per script invocation.

Each test runs one retained Sonar example in its own process from the examples
directory. Upstream-only example layouts are not part of this checkout.
"""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

# The examples run in their own processes; this process starts no engine.
pytestmark = pytest.mark.skip_global_cleanup

EXAMPLES_DIR = Path(__file__).parents[2] / "examples"

TENSORIZER_DIR = "/tmp/"
TENSORIZER_TENSORS = "/tmp/sonar/facebook/opt-125m/v1/model.tensors"

# Each case is the list of commands one test runs, in order.
EXAMPLES = [
    pytest.param(["offline_inference/chat_inference.py"], id="chat"),
    pytest.param(["offline_inference/offline_inference.py"], id="generate"),
    pytest.param(["audio/audio_example.py"], id="audio-language"),
    # Serialize then deserialize, so both commands stay in one test.
    pytest.param(
        [
            (
                "tensorize_sonar_model.py --model facebook/opt-125m serialize"
                f" --serialized-directory {TENSORIZER_DIR} --suffix v1"
            ),
            (f"tensorize_sonar_model.py --model facebook/opt-125m deserialize --path-to-tensors {TENSORIZER_TENSORS}"),
        ],
        id="features-tensorize-opt-125m",
    ),
]


@pytest.mark.parametrize("commands", EXAMPLES)
def test_offline_example(commands: list[str]):
    """Run one example's commands from the examples directory.

    Args:
        commands: Each one is an example script path, relative to the
            examples directory, followed by its arguments.

    """
    for command in commands:
        subprocess.run(
            [sys.executable, *shlex.split(command)],
            cwd=EXAMPLES_DIR,
            check=True,
        )
