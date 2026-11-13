# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite Engine project

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "unrelated",
    [
        "other_engine-1.0-cp38-abi3-linux_x86_64.whl",
        "sonar_engine-0.2.dev1+cu132-cp38-abi3-linux_x86_64.whl",
        "aphrodite_engine-0.1-cp38-abi3-linux_x86_64.whl",
    ],
)
def test_generate_nightly_index_lists_all_wheels(tmp_path: Path, unrelated: str) -> None:
    script = Path(__file__).parents[2] / ".github" / "scripts" / "generate_nightly_index.py"
    current = "sonar-0.2.dev2+cu132-cp38-abi3-linux_x86_64.whl"
    previous = "sonar-0.2.dev1+cu132-cp38-abi3-linux_x86_64.whl"
    current_url = current
    previous_url = previous.replace("+", "%2B")
    entries = tmp_path / "entries.tsv"
    entries.write_text(
        f"{current}\thttps://sonar-nightly.dphn.ai/wheels/{current_url}\tabc123\n"
        f"{previous}\thttps://sonar-nightly.dphn.ai/wheels/{previous_url}\n"
        f"{unrelated}\thttps://sonar-nightly.dphn.ai/wheels/{unrelated}\n"
    )
    output = tmp_path / "index.html"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--entry-file",
            str(entries),
            "--commit",
            "deadbeef",
            "--output",
            str(output),
        ],
        check=True,
    )

    document = output.read_text()
    canonical_current_url = current_url.replace("+", "%2B")
    assert f"https://sonar-nightly.dphn.ai/wheels/{canonical_current_url}#sha256=abc123" in document
    assert f"https://sonar-nightly.dphn.ai/wheels/{previous_url}" in document
    assert document.count(current) == 1
    assert document.count(previous) == 1
    assert unrelated not in document
