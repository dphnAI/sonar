# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _wheel(name: str, version: str, *, dependency: bool = False) -> tuple[str, bytes]:
    module = name.replace("-", "_")
    dist_info = f"{module}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    source = f'__version__ = "{version}"\n'
    if dependency:
        metadata += "Requires-Dist: sonar-installer-test-dependency==1.0\n"
        source += "import sonar_installer_test_dependency\n"
    files = {
        f"{module}/__init__.py": source,
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{dist_info}/RECORD"] = "".join(f"{path},,\n" for path in files) + f"{dist_info}/RECORD,,\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return f"{module}-{version}-py3-none-any.whl", buffer.getvalue()


@pytest.mark.parametrize(
    ("preflight_status", "resolution_status", "already_installed", "index_content"),
    [
        pytest.param(404, 404, False, "wheel", id="missing-index"),
        pytest.param(503, 503, False, "wheel", id="unavailable-index"),
        pytest.param(200, 404, False, "wheel", id="index-disappears"),
        pytest.param(200, 503, False, "wheel", id="index-fails-during-resolution"),
        pytest.param(200, 404, True, "wheel", id="cannot-reuse-unrelated-install"),
        pytest.param(200, 200, False, "wheel", id="trusted-package-public-dependency"),
        pytest.param(200, 200, True, "wheel", id="replace-unrelated-install"),
        pytest.param(200, 200, False, "empty", id="empty-index"),
        pytest.param(200, 200, False, "unrelated", id="unrelated-page"),
        pytest.param(200, 200, False, "sdist", id="source-distribution-only"),
    ],
)
def test_installer_pins_sonar_index(
    tmp_path: Path,
    preflight_status: int,
    resolution_status: int,
    already_installed: bool,
    index_content: str,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for installer integration tests")
        return
    architecture = platform.machine().lower()
    if sys.platform != "linux" or architecture not in ("x86_64", "aarch64"):
        pytest.skip("installer integration test requires a supported Linux platform")
    backend = "rocm" if architecture == "x86_64" else "cpu"
    if backend == "cpu":
        try:
            ctypes.CDLL("libnuma.so.1")
        except OSError:
            pytest.skip("CPU installer requires libnuma; tests must not install system packages")

    trusted_name, trusted_wheel = _wheel("sonar", "0.1", dependency=True)
    unrelated_name, unrelated_wheel = _wheel("sonar", "9.9")
    dependency_name, dependency_wheel = _wheel("sonar-installer-test-dependency", "1.0")
    wheels = {
        f"/files/{trusted_name}": trusted_wheel,
        f"/files/{unrelated_name}": unrelated_wheel,
        f"/files/{dependency_name}": dependency_wheel,
    }
    trusted_page = f"/whl/release/{backend}/{architecture}/simple/sonar/"
    fallback_page = "/fallback/simple/sonar/"
    dependency_page = "/fallback/simple/sonar-installer-test-dependency/"
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            requests.append(self.path)
            if self.path not in wheels:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(wheels[self.path])))
            self.end_headers()

        def do_GET(self) -> None:
            requests.append(self.path)
            if self.path == trusted_page:
                status = preflight_status if requests.count(trusted_page) == 1 else resolution_status
                if status != 200:
                    self.send_error(status)
                    return
                name = trusted_name
            elif self.path == fallback_page:
                name = unrelated_name
            elif self.path == dependency_page:
                name = dependency_name
            elif self.path in wheels:
                payload = wheels[self.path]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            else:
                self.send_error(404)
                return
            if self.path == trusted_page and index_content == "empty":
                payload = b"<html><body></body></html>"
            elif self.path == trusted_page and index_content == "unrelated":
                payload = b'<html><body><a href="/login">Sign in to access the internet</a></body></html>'
            elif self.path == trusted_page and index_content == "sdist":
                payload = b'<a href="/files/sonar-0.1.tar.gz">sonar-0.1.tar.gz</a>'
            else:
                payload = f'<a href="/files/{name}">{name}</a>'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    env = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    env.update(
        SONAR_BASE_URL=base_url,
        UV_DEFAULT_INDEX=f"{base_url}/fallback/simple",
        UV_NO_CONFIG="1",
        UV_NO_SOURCES="1",
        UV_CACHE_DIR=str(tmp_path / "uv-cache"),
        UV_PYTHON_DOWNLOADS="never",
        UV_HTTP_RETRIES="0",
        NO_PROXY="127.0.0.1,localhost",
    )
    venv = tmp_path / "venv"
    try:
        if preflight_status == 200:
            subprocess.run([uv, "venv", "--python", sys.executable, str(venv)], env=env, check=True)
        python = str(venv / "bin/python")
        if already_installed:
            unrelated = tmp_path / unrelated_name
            unrelated.write_bytes(unrelated_wheel)
            subprocess.run([uv, "pip", "install", "--python", python, str(unrelated)], env=env, check=True)
        result = subprocess.run(
            [
                "sh",
                str(Path(__file__).resolve().parents[2] / "docs/public/install.sh"),
                "--yes",
                "--backend",
                backend,
                "--channel",
                "release",
                "--skip-platform-checks",
                "--venv",
                str(venv),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if preflight_status == resolution_status == 200 and index_content == "wheel":
            assert result.returncode == 0, result.stdout + result.stderr
            installed = subprocess.check_output(
                [python, "-I", "-c", "import sonar; print(sonar.__version__)"], text=True
            ).strip()
            assert installed == "0.1"
            assert dependency_page in requests
        elif preflight_status != 200:
            assert result.returncode != 0, result.stdout + result.stderr
            assert not venv.exists()
        else:
            assert result.returncode != 0, result.stdout + result.stderr
            installed = json.loads(
                subprocess.check_output(
                    [
                        python,
                        "-I",
                        "-c",
                        (
                            "import importlib.metadata as m, json; "
                            "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"
                        ),
                    ],
                    text=True,
                )
            )
            assert installed.get("sonar") == ("9.9" if already_installed else None)
            assert not (venv / ".sonar-install.json").exists()
        assert fallback_page not in requests, result.stdout + result.stderr
        if index_content == "sdist":
            assert "/files/sonar-0.1.tar.gz" not in requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
