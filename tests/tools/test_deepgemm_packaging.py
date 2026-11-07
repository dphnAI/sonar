# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_BINDINGS = ("_C.cpython-311-x86_64-linux-gnu.so", "_C.abi3.so")


def seed_installed_package(package: Path) -> None:
    package.mkdir(parents=True)
    for filename in LEGACY_BINDINGS:
        (package / filename).write_bytes(b"old pybind binding")
    (package / "_C_extension.abi3.so").write_bytes(b"stable binding")
    (package / "user_kernel.so").write_bytes(b"keep user kernel")
    (package / "user_notes.txt").write_text("keep user notes\n")


def assert_legacy_bindings_removed(package: Path) -> None:
    assert all(not (package / filename).exists() for filename in LEGACY_BINDINGS)
    assert (package / "_C_extension.abi3.so").read_bytes() == b"stable binding"
    assert (package / "user_kernel.so").read_bytes() == b"keep user kernel"
    assert (package / "user_notes.txt").read_text() == "keep user notes\n"


@pytest.mark.parametrize("use_destdir", [False, True])
def test_cmake_install_removes_only_legacy_deepgemm_bindings(tmp_path: Path, use_destdir: bool) -> None:
    source = (REPO_ROOT / "cmake/external_projects/deepgemm.cmake").read_text()
    cleanup = source.split("install(CODE [[", 1)[1].split("]] COMPONENT _deep_gemm_C)", 1)[0]
    script = tmp_path / "cleanup.cmake"
    script.write_text(cleanup)
    prefix = tmp_path / "prefix"
    environment = os.environ.copy()
    environment.pop("DESTDIR", None)
    if use_destdir:
        environment["DESTDIR"] = str(tmp_path / "destdir")
        destination = Path(environment["DESTDIR"] + str(prefix))
    else:
        destination = prefix
    package = destination / "sonar/third_party/deep_gemm"
    seed_installed_package(package)

    subprocess.run(
        ["cmake", f"-DCMAKE_INSTALL_PREFIX={prefix}", "-P", str(script)],
        env=environment,
        check=True,
    )

    assert_legacy_bindings_removed(package)


def test_editable_copy_removes_only_legacy_deepgemm_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = ast.parse((REPO_ROOT / "setup.py").read_text())
    declaration = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "cmake_build_ext"
    )
    declaration.body = [node for node in declaration.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    module = ast.fix_missing_locations(ast.Module(body=[declaration], type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "packaging_test",
        "build_ext": type("BuildExt", (), {"run": lambda self: None}),
        "Path": Path,
        "ROOT_DIR": tmp_path,
        "os": os,
        "shutil": shutil,
        "should_bundle_tcmalloc": lambda: False,
        "_is_cuda": lambda: True,
        "_is_hip": lambda: False,
    }
    exec(compile(module, "setup.py", "exec"), namespace)
    staging = tmp_path / "build"
    (staging / "sonar/third_party/triton_kernels").mkdir(parents=True)
    staged_package = staging / "sonar/third_party/deep_gemm"
    staged_package.mkdir()
    (staged_package / "_C.py").write_text("# stable facade\n")
    (staged_package / "_C_extension.abi3.so").write_bytes(b"stable binding")
    installed_package = tmp_path / "sonar/third_party/deep_gemm"
    seed_installed_package(installed_package)
    command = namespace["cmake_build_ext"]()
    command.build_lib = str(staging)
    monkeypatch.chdir(tmp_path)

    command.run()

    assert_legacy_bindings_removed(installed_package)
    assert (installed_package / "_C.py").read_text() == "# stable facade\n"
