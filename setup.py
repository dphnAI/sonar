# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import importlib.util
import logging
import os
import subprocess
import sys
import warnings
from contextlib import suppress
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

_VERSION_OVERRIDE = os.getenv("APHRODITE_VERSION_OVERRIDE")
if _VERSION_OVERRIDE:
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = _VERSION_OVERRIDE

# Capture the SCM-derived version before setup-time generated files dirty an
# exact-tag checkout. Otherwise release Docker builds can become
# `<next>.dev0+g<sha>.d<date>` after `embed_commit_hash()` writes commit_id.py.
_SCM_VERSION = get_version(write_to="aphrodite/_version.py")


def embed_commit_hash():
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8").strip()
        short_commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], encoding="utf-8").strip()

        commit_contents = f'__commit__ = "{commit_id}"\n'
        short_commit_contents = f'__short_commit__ = "{short_commit_id}"\n'

        version_file = os.path.join(ROOT_DIR, "aphrodite", "commit_id.py")
        with open(version_file, "w", encoding="utf-8") as f:
            f.write(commit_contents)
            f.write(short_commit_contents)

    except subprocess.CalledProcessError as e:
        warnings.warn(f"Failed to get commit hash:\n{e}", RuntimeWarning, stacklevel=2)
    except Exception as e:
        warnings.warn(f"Failed to embed commit hash:\n{e}", RuntimeWarning, stacklevel=2)


embed_commit_hash()

# cannot import envs directly because it depends on aphrodite,
#  which is not installed yet
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "aphrodite", "envs.py"))

APHRODITE_TARGET_DEVICE = envs.APHRODITE_TARGET_DEVICE

if sys.platform.startswith("darwin") and APHRODITE_TARGET_DEVICE not in ("cpu", "metal"):
    logger.warning("APHRODITE_TARGET_DEVICE automatically set to `metal` due to macOS")
    APHRODITE_TARGET_DEVICE = "metal"
elif not (sys.platform.startswith("linux") or sys.platform.startswith("darwin")):
    logger.warning(
        "Aphrodite only supports Linux platform (including WSL) and MacOS."
        "Building on {}, "
        "so Aphrodite may not be able to run correctly",
        sys.platform,
    )
    APHRODITE_TARGET_DEVICE = "empty"
elif (
    sys.platform.startswith("linux")
    and torch.version.cuda is None
    and os.getenv("APHRODITE_TARGET_DEVICE") is None
    and torch.version.hip is None
):
    # if cuda or hip is not available and APHRODITE_TARGET_DEVICE is not set,
    # fallback to cpu
    APHRODITE_TARGET_DEVICE = "cpu"

MAIN_CUDA_VERSION = "12.8"


def _get_available_memory_bytes() -> int | None:
    """Return available system memory in bytes, or None if unknown.

    Tries multiple strategies in order:
    - psutil (if available)
    - POSIX sysconf with SC_AVPHYS_PAGES
    - /proc/meminfo on Linux (MemAvailable)
    - vm_stat on macOS (free + inactive pages)
    """
    # Try psutil if available
    with suppress(Exception):
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)

    # Try POSIX sysconf for available pages
    with suppress(Exception):
        page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[arg-type]
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")  # type: ignore[arg-type]
        return int(page_size) * int(avail_pages)

    # Linux fallback: /proc/meminfo
    with suppress(Exception):
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        # Value is in kB
                        return int(parts[1]) * 1024

    # macOS fallback: vm_stat
    with suppress(Exception):
        if sys.platform.startswith("darwin"):
            out = subprocess.check_output(["vm_stat"], encoding="utf-8")
            page_size_bytes = 4096
            for line in out.splitlines():
                if "page size of" in line and "bytes" in line:
                    # e.g., "Mach VM Stats: (page size of 16384 bytes)"
                    with suppress(Exception):
                        page_size_bytes = int(line.split("page size of")[1].split("bytes")[0].strip())
                    break
            pages_free = 0
            pages_inactive = 0
            for line in out.splitlines():
                if line.strip().startswith("Pages free"):
                    pages_free = int(line.split(":")[1].strip().strip(". "))
                elif line.strip().startswith("Pages inactive"):
                    pages_inactive = int(line.split(":")[1].strip().strip(". "))
            return (pages_free + pages_inactive) * page_size_bytes

    return None


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(int(os.getenv("APHRODITE_DISABLE_SCCACHE", "0")))


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_url_available(url: str) -> bool:
    from urllib.request import urlopen

    status = None
    try:
        with urlopen(url) as f:
            status = f.status
    except Exception:
        return False
    return status == 200


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        py_limited_api = kwa.pop("py_limited_api", True)
        super().__init__(name, sources=[], py_limited_api=py_limited_api, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or computed from available RAM (6 GiB per job),
        # falling back to the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            available_bytes = _get_available_memory_bytes()
            if available_bytes is not None and available_bytes > 0:
                available_gib = max(0, available_bytes // (1024**3))
                # Heuristic: 8 GiB per job
                num_jobs = max(1, int(available_gib // 8))
                logger.info(
                    "RAM heuristic: ~%d GiB avail -> num_jobs=%d (8 GiB/job). If you think this is too low or too high,"
                    " set MAX_JOBS to a higher value.",
                    available_gib,
                    num_jobs,
                )
            else:
                try:
                    # os.sched_getaffinity() not always available; fallback to
                    # os.cpu_count() when needed.
                    num_jobs = len(os.sched_getaffinity(0))
                    logger.info("CPU heuristic: %d jobs (RAM unknown).", num_jobs)
                except AttributeError:
                    num_jobs = os.cpu_count()
                    logger.info("CPU heuristic: os.cpu_count()=%d (RAM unk).", num_jobs)

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = envs.NVCC_THREADS
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info("Using NVCC_THREADS=%d as the number of nvcc threads.", nvcc_threads)
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DAPHRODITE_TARGET_DEVICE={}".format(APHRODITE_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ["-DAPHRODITE_PYTHON_EXECUTABLE={}".format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ["-DAPHRODITE_PYTHON_PATH={}".format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ["-DNVCC_THREADS={}".format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        # Make sure we use the nvcc from CUDA_HOME
        if _is_cuda():
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]

        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args += other_cmake_args.split()

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            if s.endswith("._paged_ops"):
                return "_paged_ops"
            return s.removeprefix("aphrodite.").removeprefix("vllm_flash_attn.").removeprefix("aphrodite_flash_attn.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            # We assume only the final component of extension prefix is added by
            # CMake, this is currently true for current extensions but may not
            # always be the case.
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake",
                "--install",
                ".",
                "--prefix",
                prefix,
                "--component",
                target_name(ext.name),
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # Run the standard build_ext command to compile the extensions
        super().run()

        # Copy aphrodite/vllm_flash_attn/**/*.py from self.build_lib to the
        # source tree so editable installs can import the packaged Python
        # wrappers alongside the built extension modules.
        import glob
        import shutil

        files = glob.glob(
            os.path.join(self.build_lib, "aphrodite", "vllm_flash_attn", "**", "*.py"),
            recursive=True,
        )
        for file in files:
            dst_file = os.path.join(
                "aphrodite/vllm_flash_attn",
                file.split("aphrodite/vllm_flash_attn/")[-1],
            )
            print(f"Copying {file} to {dst_file}")
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            self.copy_file(file, dst_file)

        if _is_cuda() or _is_hip():
            triton_kernels_build = os.path.join(self.build_lib, "aphrodite", "third_party", "triton_kernels")
            if os.path.exists(triton_kernels_build):
                print(f"Copying {triton_kernels_build} to aphrodite/third_party/triton_kernels")
                shutil.copytree(
                    triton_kernels_build,
                    "aphrodite/third_party/triton_kernels",
                    dirs_exist_ok=True,
                )

        if _is_cuda():
            deep_gemm_build = os.path.join(self.build_lib, "aphrodite", "third_party", "deep_gemm")
            if os.path.exists(deep_gemm_build):
                print(f"Copying {deep_gemm_build} to aphrodite/third_party/deep_gemm")
                shutil.copytree(
                    deep_gemm_build,
                    "aphrodite/third_party/deep_gemm",
                    dirs_exist_ok=True,
                )


def _is_hpu() -> bool:
    # if APHRODITE_TARGET_DEVICE env var was set explicitly, skip autodetection
    if os.getenv("APHRODITE_TARGET_DEVICE", None) == APHRODITE_TARGET_DEVICE:
        return APHRODITE_TARGET_DEVICE == "hpu"

    # if APHRODITE_TARGET_DEVICE was not set explicitly, check if hl-smi works;
    # otherwise, check if habanalabs driver is loaded
    is_hpu_available = False
    try:
        out = subprocess.run(["hl-smi"], capture_output=True, check=True)
        is_hpu_available = out.returncode == 0
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        if sys.platform.startswith("linux"):
            try:
                output = subprocess.check_output("lsmod | grep habanalabs | wc -l", shell=True)
                is_hpu_available = int(output) > 0
            except (
                ValueError,
                FileNotFoundError,
                PermissionError,
                subprocess.CalledProcessError,
            ):
                pass
    return is_hpu_available


def _no_device() -> bool:
    return APHRODITE_TARGET_DEVICE == "empty"


def _is_windows() -> bool:
    return APHRODITE_TARGET_DEVICE == "windows"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return APHRODITE_TARGET_DEVICE == "cuda" and has_cuda and not (_is_tpu() or _is_hpu())


def _is_hip() -> bool:
    return (APHRODITE_TARGET_DEVICE == "cuda" or APHRODITE_TARGET_DEVICE == "rocm") and torch.version.hip is not None


def _is_tpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "tpu"


def _is_cpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "cpu"


def _is_xpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "xpu"


def _is_metal() -> bool:
    return APHRODITE_TARGET_DEVICE == "metal"


def _build_custom_ops() -> bool:
    # Skip building custom ops if using precompiled binaries
    if envs.APHRODITE_USE_PRECOMPILED:
        return False
    return _is_cuda() or _is_hip() or _is_cpu() or _is_metal()


def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    # see https://github.com/ROCm/rocm-core/blob/d11f5c20d500f729c393680a01fa902ebf92094b/rocm_version.cpp#L21
    try:
        librocm_core_file = Path(ROCM_HOME) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(librocm_core_file)
        VerErrors = ctypes.c_uint32
        get_rocm_core_version = librocm_core.getROCmVersion
        get_rocm_core_version.restype = VerErrors
        get_rocm_core_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()

        if get_rocm_core_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)) == 0:
            return f"{major.value}.{minor.value}.{patch.value}"
        return None
    except Exception:
        return None


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_gaudi_sw_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run(
        "hl-smi",
        shell=True,
        text=True,
        capture_output=True,
        env={"ENABLE_CONSOLE": "true"},
    )
    if output.returncode == 0 and output.stdout:
        return output.stdout.split("\n")[2].replace(" ", "").split(":")[1][:-1].split("-")[0]
    return "0.0.0"  # when hl-smi is not available


def get_aphrodite_version() -> str:
    # Allow overriding the version. This is useful to build platform-specific
    # wheels (e.g. CPU, TPU) without modifying the source.
    version = _SCM_VERSION
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _no_device():
        if envs.APHRODITE_TARGET_DEVICE == "empty":
            version += f"{sep}empty"
    elif _is_cuda():
        if envs.APHRODITE_USE_PRECOMPILED:
            version += f"{sep}precompiled"
        else:
            cuda_version = str(get_nvcc_cuda_version())
            if cuda_version != envs.APHRODITE_MAIN_CUDA_VERSION:
                cuda_version_str = cuda_version.replace(".", "")[:3]
                # skip this for source tarball, required for pypi
                if "sdist" not in sys.argv:
                    version += f"{sep}cu{cuda_version_str}"
    elif _is_hip():
        # Get the Rocm Version
        rocm_version = get_rocm_version() or torch.version.hip
        if rocm_version and rocm_version != envs.APHRODITE_MAIN_CUDA_VERSION:
            version += f"{sep}rocm{rocm_version.replace('.', '')[:3]}"
    elif _is_hpu():
        # Get the Intel Gaudi Software Suite version
        gaudi_sw_version = str(get_gaudi_sw_version())
        if gaudi_sw_version != envs.APHRODITE_MAIN_CUDA_VERSION:
            gaudi_sw_version = gaudi_sw_version.replace(".", "")[:3]
            version += f"{sep}gaudi{gaudi_sw_version}"
    elif _is_tpu():
        version += f"{sep}tpu"
    elif _is_cpu():
        # Check the local APHRODITE_TARGET_DEVICE (may be set by
        # auto-detect above), not envs.APHRODITE_TARGET_DEVICE, so CPU-only
        # hosts still get `+cpu`.
        if APHRODITE_TARGET_DEVICE == "cpu":
            version += f"{sep}cpu"
    elif _is_xpu():
        version += f"{sep}xpu"
    elif _is_metal():
        version += f"{sep}metal"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith("--") and not line.startswith("#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements

    if _no_device():
        requirements = _read_requirements("common.txt")
    elif _is_cuda():
        requirements = _read_requirements("cuda.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if "aphrodite-flash-attn" in req and cuda_major != "12":
                # aphrodite-flash-attn is built only for CUDA 12.x.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
        requirements = modified_requirements

    elif _is_hip():
        requirements = _read_requirements("rocm.txt")
    elif _is_hpu():
        requirements = _read_requirements("hpu.txt")
    elif _is_tpu():
        requirements = _read_requirements("tpu.txt")
    elif _is_cpu():
        requirements = _read_requirements("cpu.txt")
    elif _is_xpu():
        requirements = _read_requirements("xpu.txt")
    elif _is_metal():
        requirements = _read_requirements("metal.txt")
    else:
        raise ValueError("Unsupported platform, please use CUDA, ROCm, Metal, or CPU.")

    return requirements


ext_modules = []
if _build_custom_ops():
    if _is_metal():
        ext_modules.append(
            CMakeExtension(
                name="aphrodite.metal.metal._paged_ops",
                cmake_lists_dir=".",
                py_limited_api=False,
            )
        )

    if _is_cuda() or _is_hip():
        ext_modules.append(CMakeExtension(name="aphrodite._moe_C", cmake_lists_dir="."))
        # Optional: this target vendors the triton_kernels Python package.
        ext_modules.append(CMakeExtension(name="aphrodite.triton_kernels", cmake_lists_dir=".", optional=True))

    if _is_hip():
        ext_modules.append(CMakeExtension(name="aphrodite._rocm_C", cmake_lists_dir="."))

    if _is_cuda():
        ext_modules.append(CMakeExtension(name="aphrodite._C_stable_libtorch", cmake_lists_dir="."))

        disable_flash_attn = os.environ.get("APHRODITE_DISABLE_FLASH_ATTN_COMPILE", "0").strip().lower() in (
            "1",
            "true",
        )
        if not disable_flash_attn:
            ext_modules.append(CMakeExtension(name="aphrodite.vllm_flash_attn._vllm_fa2_C", cmake_lists_dir="."))
            if envs.APHRODITE_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.3"):
                ext_modules.append(
                    CMakeExtension(
                        name="aphrodite.vllm_flash_attn._vllm_fa3_C",
                        cmake_lists_dir=".",
                    )
                )
            ext_modules.append(
                CMakeExtension(
                    name="aphrodite.vllm_flash_attn._vllm_fa4_cutedsl_C",
                    cmake_lists_dir=".",
                    optional=True,
                )
            )

        if envs.APHRODITE_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.3"):
            ext_modules.append(CMakeExtension(name="aphrodite._flashmla_C", cmake_lists_dir=".", optional=True))
            ext_modules.append(
                CMakeExtension(
                    name="aphrodite._flashmla_extension_C",
                    cmake_lists_dir=".",
                    optional=True,
                )
            )
            ext_modules.append(CMakeExtension(name="aphrodite._deep_gemm_C", cmake_lists_dir=".", optional=True))
        ext_modules.append(CMakeExtension(name="aphrodite.cumem_allocator", cmake_lists_dir="."))

    if not _is_metal():
        ext_modules.append(CMakeExtension(name="aphrodite._C", cmake_lists_dir="."))

if _no_device():
    ext_modules = []

package_data = {
    "aphrodite": [
        "endpoints/kobold/klite.embd",
        "quantization/hadamard.safetensors",
        "py.typed",
        "modeling/layers/fused_moe/configs/*.json",
        "third_party/deep_gemm/include/**/*.cuh",
        "third_party/deep_gemm/include/**/*.h",
        "third_party/deep_gemm/include/**/*.hpp",
    ]
}

setup(
    # static metadata should rather go to pyproject.toml
    name="aphrodite-engine",
    version=get_aphrodite_version(),
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "matplotlib", "seaborn", "datasets"],
        "tensorizer": ["tensorizer==2.10.1"],
        "fastsafetensors": ["fastsafetensors >= 0.1.10"],
        "runai": ["runai-model-streamer[s3,gcs] >= 0.15.0"],
        "audio": [
            "librosa",
            "soundfile",
            "mistral_common[audio]",
        ],  # Required for audio processing
        "video": [],  # Kept for backwards compatibility
        # FlashInfer should be updated together with the Dockerfile
        "flashinfer": [],  # Kept for backwards compatibility
        # Optional deps for AMD FP4 quantization support
        "petit-kernel": ["petit-kernel"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext} if ext_modules else {},
    package_data=package_data,
)
