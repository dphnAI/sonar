# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

DEFAULT_WHEEL_BASE_URL = "https://sonar-nightly.dphn.ai"
UPSTREAM_MAIN_API = "https://api.github.com/repos/dphnAI/sonar/commits/main"
UPSTREAM_GIT_URL = "https://github.com/dphnAI/sonar.git"
MANIFEST_PATH = Path("aphrodite/.precompiled-wheel-manifest.json")
PACKAGE_NAME = "aphrodite-engine"

_HEX_DIGITS = frozenset("0123456789abcdef")
_GENERATED_TREE_PREFIXES = (
    "aphrodite/third_party/triton_kernels/",
    "aphrodite/third_party/flashmla/",
    "aphrodite/third_party/deep_gemm/",
    "aphrodite/third_party/fmha_sm100/",
    "aphrodite/third_party/tml_fa4/",
    "aphrodite/vllm_flash_attn/",
)
_TRACKED_FLASH_ATTN_FILES = {
    "aphrodite/vllm_flash_attn/__init__.py",
    "aphrodite/vllm_flash_attn/flash_attn_interface.py",
}


def _is_full_commit(value: str) -> bool:
    return len(value) == 40 and all(character in _HEX_DIGITS for character in value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX_DIGITS for character in value.lower())


class PrecompiledWheelError(RuntimeError):
    """Raised when a compatible precompiled wheel cannot be used safely."""


@dataclass(frozen=True)
class WheelCandidate:
    url_or_path: str
    filename: str
    sha256: str | None = None


class _WheelLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, "".join(self._text).strip()))
            self._href = None
            self._text = []


def _run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _github_main_commit() -> str:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "aphrodite-build"}
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(UPSTREAM_MAIN_API, headers=headers), timeout=30) as response:
        commit = json.loads(response.read().decode())["sha"].lower()
    if not _is_full_commit(commit):
        raise PrecompiledWheelError(f"GitHub returned an invalid main commit: {commit!r}")
    return commit


def get_base_commit_in_main(root: Path) -> str:
    """Return the current checkout's merge-base with upstream main."""
    try:
        head = _run_git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrecompiledWheelError(
            "Cannot determine the checkout commit. Set "
            "APHRODITE_PRECOMPILED_WHEEL_COMMIT or "
            "APHRODITE_PRECOMPILED_WHEEL_LOCATION explicitly."
        ) from exc

    main_commit: str | None = None
    try:
        candidate = _run_git(root, "rev-parse", "refs/remotes/origin/main")
        if _is_full_commit(candidate):
            main_commit = candidate
    except (OSError, subprocess.CalledProcessError):
        pass

    if main_commit is None:
        try:
            main_commit = _github_main_commit()
            try:
                _run_git(root, "cat-file", "-e", f"{main_commit}^{{commit}}")
            except subprocess.CalledProcessError:
                subprocess.check_call(
                    ["git", "-C", str(root), "fetch", "--quiet", UPSTREAM_GIT_URL, "main"],
                )
        except Exception as exc:
            raise PrecompiledWheelError(
                "Cannot resolve upstream main. Fetch main or set APHRODITE_PRECOMPILED_WHEEL_COMMIT explicitly."
            ) from exc

    try:
        base = _run_git(root, "merge-base", head, main_commit)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrecompiledWheelError(f"Cannot find a merge-base between HEAD and upstream main {main_commit}.") from exc
    if not _is_full_commit(base):
        raise PrecompiledWheelError(f"git merge-base returned an invalid commit: {base!r}")
    return base


def _validate_host() -> None:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise PrecompiledWheelError(
            "Hosted Aphrodite precompiled wheels currently support Linux x86_64 CUDA only. "
            "Use APHRODITE_PRECOMPILED_WHEEL_LOCATION for another compatible wheel, "
            "or build native extensions locally."
        )


def _candidate_from_location(location: str) -> WheelCandidate:
    parsed = urlparse(location)
    path = parsed.path if parsed.scheme else location
    filename = unquote(Path(path).name)
    if not filename.endswith(".whl"):
        raise PrecompiledWheelError(f"APHRODITE_PRECOMPILED_WHEEL_LOCATION must identify a .whl file, got {location!r}")
    digest = parse_qs(parsed.fragment).get("sha256", [None])[0]
    if digest is not None and not _is_sha256(digest):
        raise PrecompiledWheelError(f"Invalid sha256 fragment in wheel location: {digest!r}")
    return WheelCandidate(location, filename, digest.lower() if digest else None)


def _compatible_candidate(index_url: str, html: str) -> WheelCandidate:
    parser = _WheelLinkParser()
    parser.feed(html)
    supported_tags = set(sys_tags())
    available: list[str] = []

    for href, link_text in parser.links:
        filename = unquote(Path(urlparse(href).path).name)
        if not filename.endswith(".whl"):
            filename = unquote(link_text)
        if not filename.endswith(".whl"):
            continue
        available.append(filename)
        try:
            name, _, _, tags = parse_wheel_filename(filename)
        except ValueError:
            continue
        if canonicalize_name(name) != canonicalize_name(PACKAGE_NAME):
            continue
        if not supported_tags.intersection(tags):
            continue
        fragment = parse_qs(urlparse(href).fragment)
        digest = fragment.get("sha256", [None])[0]
        if digest is not None and not _is_sha256(digest):
            raise PrecompiledWheelError(f"Invalid sha256 for {filename}: {digest!r}")
        return WheelCandidate(
            urljoin(index_url, href),
            filename,
            digest.lower() if digest else None,
        )

    raise PrecompiledWheelError(
        f"No compatible aphrodite-engine wheel was found at {index_url}. Available wheels: {available or 'none'}"
    )


def determine_wheel(
    root: Path,
    *,
    location: str | None = None,
    commit: str | None = None,
    base_url: str = DEFAULT_WHEEL_BASE_URL,
) -> WheelCandidate:
    """Resolve an explicit or commit-matched precompiled wheel."""
    if location:
        print(f"Using user-specified precompiled wheel: {location}")
        return _candidate_from_location(location)

    selected_commit = (commit or "").strip().lower()
    if not selected_commit:
        selected_commit = get_base_commit_in_main(root)
    elif selected_commit != "nightly" and not _is_full_commit(selected_commit):
        raise PrecompiledWheelError(
            "APHRODITE_PRECOMPILED_WHEEL_COMMIT must be a full 40-character "
            f"commit SHA or 'nightly', got {selected_commit!r}"
        )
    _validate_host()

    if selected_commit == "nightly":
        index_url = f"{base_url.rstrip('/')}/nightly/{PACKAGE_NAME}/index.html"
    else:
        index_url = f"{base_url.rstrip('/')}/commits/{selected_commit}/{PACKAGE_NAME}/index.html"

    print(f"Fetching Aphrodite precompiled wheel index: {index_url}")
    try:
        with urlopen(Request(index_url, headers={"User-Agent": "aphrodite-build"}), timeout=60) as response:
            html = response.read().decode()
    except Exception as exc:
        raise PrecompiledWheelError(
            f"Precompiled wheel index is unavailable for {selected_commit}: {index_url}. "
            "Wait for its wheel build, set APHRODITE_PRECOMPILED_WHEEL_COMMIT=nightly, "
            "provide APHRODITE_PRECOMPILED_WHEEL_LOCATION, or build locally."
        ) from exc

    candidate = _compatible_candidate(index_url, html)
    print(f"Using precompiled wheel: {candidate.url_or_path}")
    return candidate


def _should_extract(filename: str, *, extensions: bool, rust: bool) -> bool:
    if filename in _TRACKED_FLASH_ATTN_FILES:
        return False
    if extensions and filename.startswith("aphrodite/") and filename.endswith(".so"):
        return True
    if extensions and filename.startswith("aphrodite/libs/") and ".so." in PurePosixPath(filename).name:
        # CPU wheels bundle versioned tcmalloc libraries alongside extensions.
        return True
    rust_extension = (
        filename.startswith("aphrodite/_rust_")
        and filename.endswith(".so")
        and "/" not in filename.removeprefix("aphrodite/")
    )
    if rust and (filename == "aphrodite/aphrodite-rs" or rust_extension):
        return True
    return extensions and filename.startswith(_GENERATED_TREE_PREFIXES)


def _safe_member_path(root: Path, filename: str) -> Path:
    member = PurePosixPath(filename)
    if member.is_absolute() or ".." in member.parts:
        raise PrecompiledWheelError(f"Unsafe path in precompiled wheel: {filename!r}")
    target = root.joinpath(*member.parts)
    if not target.is_relative_to(root):
        raise PrecompiledWheelError(f"Unsafe path in precompiled wheel: {filename!r}")
    return target


def _package_data_patch(paths: list[str]) -> dict[str, list[str]]:
    patch: dict[str, list[str]] = {}
    for filename in paths:
        package = str(PurePosixPath(filename).parent).replace("/", ".")
        patch.setdefault(package, []).append(PurePosixPath(filename).name)
    return patch


def _load_manifest(root: Path) -> dict | None:
    path = root / MANIFEST_PATH
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("paths"), list):
        return None
    try:
        for filename in value["paths"]:
            if not isinstance(filename, str) or not _should_extract(filename, extensions=True, rust=True):
                return None
            _safe_member_path(root, filename)
    except PrecompiledWheelError:
        return None
    return value


def _remove_previous_artifacts(root: Path, manifest: dict | None) -> None:
    if manifest is None:
        return
    for filename in manifest["paths"]:
        if not isinstance(filename, str) or not _should_extract(filename, extensions=True, rust=True):
            continue
        target = _safe_member_path(root, filename)
        if target.is_file() or target.is_symlink():
            target.unlink()


def _remove_legacy_deepgemm_bindings(root: Path, paths: list[str]) -> None:
    package = "aphrodite/third_party/deep_gemm"
    if f"{package}/_C.py" not in paths or f"{package}/_C_extension.abi3.so" not in paths:
        return
    for pattern in ("_C.cpython-*.so", "_C.abi3.so"):
        for stale_binding in (root / package).glob(pattern):
            if stale_binding.is_file() or stale_binding.is_symlink():
                stale_binding.unlink()


def extract_precompiled_wheel(
    root: Path,
    candidate: WheelCandidate,
    *,
    extract_extensions: bool = True,
    extract_rust: bool = True,
) -> dict[str, list[str]]:
    """Download, verify, and extract approved artifacts into the source tree."""
    root = root.resolve()
    previous = _load_manifest(root)
    if (
        candidate.sha256
        and previous
        and previous.get("url") == candidate.url_or_path
        and previous.get("sha256") == candidate.sha256
        and all((root / path).is_file() for path in previous["paths"])
    ):
        print(f"Reusing {len(previous['paths'])} previously extracted precompiled artifacts")
        _remove_legacy_deepgemm_bindings(root, previous["paths"])
        return _package_data_patch(previous["paths"])

    with tempfile.TemporaryDirectory(prefix="aphrodite-precompiled-") as temp_dir:
        wheel_path = Path(temp_dir) / candidate.filename
        if Path(candidate.url_or_path).is_file():
            shutil.copyfile(candidate.url_or_path, wheel_path)
        else:
            print(f"Downloading {candidate.url_or_path}")
            request = Request(
                candidate.url_or_path,
                headers={"User-Agent": "aphrodite-build"},
            )
            with urlopen(request, timeout=300) as response, wheel_path.open("wb") as destination:
                shutil.copyfileobj(response, destination)

        hasher = hashlib.sha256()
        with wheel_path.open("rb") as wheel_file:
            while chunk := wheel_file.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if candidate.sha256 and digest != candidate.sha256:
            raise PrecompiledWheelError(
                f"SHA-256 mismatch for {candidate.filename}: expected {candidate.sha256}, got {digest}"
            )

        staged = Path(temp_dir) / "extracted"
        selected: list[tuple[zipfile.ZipInfo, Path]] = []
        with zipfile.ZipFile(wheel_path) as wheel:
            for member in wheel.infolist():
                if member.is_dir() or not _should_extract(
                    member.filename,
                    extensions=extract_extensions,
                    rust=extract_rust,
                ):
                    continue
                target = _safe_member_path(root, member.filename)
                staged_target = _safe_member_path(staged, member.filename)
                selected.append((member, target))
                staged_target.parent.mkdir(parents=True, exist_ok=True)
                with wheel.open(member) as source, staged_target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                mode = member.external_attr >> 16
                if mode:
                    staged_target.chmod(stat.S_IMODE(mode))

        if not selected:
            raise PrecompiledWheelError(f"{candidate.filename} contained no supported precompiled artifacts")

        _remove_previous_artifacts(root, previous)
        paths: list[str] = []
        for member, target in selected:
            staged_source = _safe_member_path(staged, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_source, target)
            paths.append(member.filename)
            print(f"[extract] {member.filename}")
        _remove_legacy_deepgemm_bindings(root, paths)

    manifest_path = root / MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "url": candidate.url_or_path,
                "sha256": candidate.sha256 or digest,
                "paths": paths,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return _package_data_patch(paths)
