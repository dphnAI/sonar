# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite Engine project

import subprocess
import sys
from pathlib import Path


def test_generate_nightly_index_lists_all_wheels(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / ".github" / "scripts" / "generate_nightly_index.py"
    current = "aphrodite_engine-0.2.dev2+cu130-cp38-abi3-linux_x86_64.whl"
    previous = "aphrodite_engine-0.2.dev1+cu130-cp38-abi3-linux_x86_64.whl"
    current_url = current
    previous_url = previous.replace("+", "%2B")
    entries = tmp_path / "entries.tsv"
    entries.write_text(
        f"{current}\thttps://sonar-nightly.dphn.ai/wheels/{current_url}\tabc123\n"
        f"{previous}\thttps://sonar-nightly.dphn.ai/wheels/{previous_url}\n"
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
    assert "Latest build from deadbeef" in document
    assert "Available builds" in document
    assert "2 wheels" in document
    assert "uv pip install aphrodite-engine --index-url https://sonar.dphn.ai/nightly" in document
    assert 'data-requires-python="&gt;=3.10,&lt;3.15"' in document
    assert "0.2.dev2+cu130" in document
    assert "linux x86_64" in document
    assert 'id="pagination"' in document
    assert "const pageSize = 50" in document
