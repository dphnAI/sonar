# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
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


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

PRECOMPILED_RUST_FRONTEND_PATH = ROOT_DIR / "aphrodite" / "aphrodite-rs"
# cannot import envs directly because it depends on aphrodite,
#  which is not installed yet
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "aphrodite", "envs.py"))

APHRODITE_TARGET_DEVICE = envs.APHRODITE_TARGET_DEVICE

# Skips all native builds (CMake/CUDA and Rust); binaries must already exist in the
# source tree — nothing is downloaded.
APHRODITE_USE_PRECOMPILED = envs.APHRODITE_USE_PRECOMPILED


def should_require_rust_frontend() -> bool:
    value = os.getenv("APHRODITE_REQUIRE_RUST_FRONTEND", "")
    return value.lower() not in ("", "0", "false", "no")


def should_use_precompiled_rust() -> bool:
    value = os.getenv("APHRODITE_USE_PRECOMPILED_RUST", "")
    return value.lower() in ("1", "true", "yes")


def get_precompiled_rust_extension_paths() -> list[Path]:
    return sorted((ROOT_DIR / "aphrodite").glob("_rust_*.so"))


if sys.platform.startswith("darwin") and APHRODITE_TARGET_DEVICE not in ("cpu", "metal"):
    logger.warning("APHRODITE_TARGET_DEVICE automatically set to `metal` due to macOS")
    APHRODITE_TARGET_DEVICE = "metal"
elif not (sys.platform.startswith("linux") or sys.platform.startswith("darwin")):
    logger.warning(
        "Aphrodite only supports Linux platform (including WSL) and MacOS."
        "Building on %s, "
        "so Aphrodite may not be able to run correctly",
        sys.platform,
    )
    APHRODITE_TARGET_DEVICE = "empty"
elif sys.platform.startswith("linux") and os.getenv("APHRODITE_TARGET_DEVICE") is None:
    if torch.version.hip is not None:
        APHRODITE_TARGET_DEVICE = "rocm"
        logger.info("Auto-detected ROCm")
    elif torch.version.xpu is not None:
        APHRODITE_TARGET_DEVICE = "xpu"
        logger.info("Auto-detected XPU")
    elif torch.version.cuda is not None:
        APHRODITE_TARGET_DEVICE = "cuda"
        logger.info("Auto-detected CUDA")
    else:
        APHRODITE_TARGET_DEVICE = "cpu"


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(int(os.getenv("APHRODITE_DISABLE_SCCACHE", "0")))


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_freethreaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def should_bundle_tcmalloc() -> bool:
    import platform

    return (
        APHRODITE_TARGET_DEVICE == "cpu"
        and sys.platform.startswith("linux")
        and platform.machine() in ("aarch64", "x86_64")
    )


def find_tcmalloc() -> Path | None:
    try:
        # get all shared libs the dynamic loader knows about
        output = subprocess.check_output(
            ["ldconfig", "-p"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    # search for libtcmalloc and libtcmalloc_minimal
    for library_pattern in (
        r"\blibtcmalloc_minimal\.so\.(\d+)\b",
        r"\blibtcmalloc\.so\.(\d+)\b",
    ):
        candidates: list[tuple[int, Path]] = []
        for line in output.splitlines():
            match = re.search(library_pattern, line)
            if match is None or "=>" not in line:
                continue
            candidate = Path(line.split("=>")[1].strip())
            if candidate.exists():
                candidates.append((int(match.group(1)), candidate))

        if candidates:
            # if multiple candidates are found, pick the one with the highest
            # version number
            return max(candidates, key=lambda item: item[0])[1]

    return None


def bundle_tcmalloc(build_lib: str) -> None:
    tcmalloc_library = find_tcmalloc()
    if tcmalloc_library is None:
        logger.warning(
            "Failed to locate tcmalloc. For best performance, "
            "please install tcmalloc (e.g. `sudo apt-get "
            "install -y --no-install-recommends libtcmalloc-minimal4`)"
        )
        return

    bundle_dir = os.path.join(build_lib, "aphrodite", "libs")
    os.makedirs(bundle_dir, exist_ok=True)
    bundle_path = os.path.join(bundle_dir, tcmalloc_library.name)
    shutil.copy2(tcmalloc_library, bundle_path)
    logger.info("Bundled tcmalloc into wheel: %s", bundle_path)


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        # Default to the stable/abi3 API, but let callers opt out (e.g. the
        # nanobind _paged_ops Metal kernel builds a version-specific module).
        kwa.setdefault("py_limited_api", not is_freethreaded())
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if _is_cuda() and CUDA_HOME is not None:
            try:
                nvcc_version = get_nvcc_cuda_version()
                if nvcc_version >= Version("11.2"):
                    # `nvcc_threads` is either the value of the NVCC_THREADS
                    # environment variable (if defined) or 1.
                    # when it is set, we reduce `num_jobs` to avoid
                    # overloading the system.
                    nvcc_threads = envs.NVCC_THREADS
                    if nvcc_threads is not None:
                        nvcc_threads = int(nvcc_threads)
                        logger.info(
                            "Using NVCC_THREADS=%d as the number of nvcc threads.",
                            nvcc_threads,
                        )
                    else:
                        nvcc_threads = 1
                    num_jobs = max(1, num_jobs // nvcc_threads)
            except Exception as e:
                logger.warning("Failed to get NVCC version: %s", e)

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

        #
        # Setup parallelism and build tool
        #
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
        if _is_cuda() and CUDA_HOME is not None:
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]
        elif _is_hip() and ROCM_HOME is not None:
            cmake_args += [f"-DROCM_PATH={ROCM_HOME}"]

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
            return s.removeprefix("aphrodite.").removeprefix("vllm_flash_attn.")

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
        # First, run the standard build_ext command to compile the extensions
        super().run()

        # bundle tcmalloc into CPU wheels for best OOB perf
        if should_bundle_tcmalloc():
            bundle_tcmalloc(self.build_lib)

        # copy aphrodite/vllm_flash_attn/**/*.py from self.build_lib to current
        # directory so that they can be included in the editable build
        import glob

        files = glob.glob(
            os.path.join(self.build_lib, "aphrodite", "vllm_flash_attn", "**", "*.py"),
            recursive=True,
        )
        for file in files:
            dst_file = os.path.join("aphrodite/vllm_flash_attn", file.split("aphrodite/vllm_flash_attn/")[-1])
            print(f"Copying {file} to {dst_file}")
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            self.copy_file(file, dst_file)

        if _is_cuda() or _is_hip():
            # copy aphrodite/third_party/triton_kernels/**/*.py from self.build_lib
            # to current directory so that they can be included in the editable
            # build
            print(
                f"Copying {self.build_lib}/aphrodite/third_party/triton_kernels to aphrodite/third_party/triton_kernels"
            )
            shutil.copytree(
                f"{self.build_lib}/aphrodite/third_party/triton_kernels",
                "aphrodite/third_party/triton_kernels",
                dirs_exist_ok=True,
            )

        if _is_cuda():
            # copy vendored deep_gemm package from build_lib to source tree
            # for editable installs
            deep_gemm_build = os.path.join(self.build_lib, "aphrodite", "third_party", "deep_gemm")
            if os.path.exists(deep_gemm_build):
                print(f"Copying {deep_gemm_build} to aphrodite/third_party/deep_gemm")
                shutil.copytree(
                    deep_gemm_build,
                    "aphrodite/third_party/deep_gemm",
                    dirs_exist_ok=True,
                )

            # copy vendored fmha_sm100 package from build_lib to source tree
            # for editable installs
            fmha_sm100_build = os.path.join(self.build_lib, "aphrodite", "third_party", "fmha_sm100")
            if os.path.exists(fmha_sm100_build):
                print(f"Copying {fmha_sm100_build} to aphrodite/third_party/fmha_sm100")
                shutil.copytree(
                    fmha_sm100_build,
                    "aphrodite/third_party/fmha_sm100",
                    dirs_exist_ok=True,
                )

            # copy vendored tml-fa4 package from build_lib to source tree
            # for editable installs
            tml_fa4_build = os.path.join(self.build_lib, "aphrodite", "third_party", "tml_fa4")
            if os.path.exists(tml_fa4_build):
                print(f"Copying {tml_fa4_build} to aphrodite/third_party/tml_fa4")
                shutil.copytree(
                    tml_fa4_build,
                    "aphrodite/third_party/tml_fa4",
                    dirs_exist_ok=True,
                )


def _no_device() -> bool:
    return APHRODITE_TARGET_DEVICE == "empty"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return APHRODITE_TARGET_DEVICE == "cuda" and has_cuda and not _is_tpu()


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
    return _is_cuda() or _is_hip() or _is_metal()


def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    # see https://github.com/ROCm/rocm-core/blob/d11f5c20d500f729c393680a01fa902ebf92094b/rocm_version.cpp#L21
    try:
        if ROCM_HOME is None:
            return None
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


def get_aphrodite_version() -> str:
    # Allow overriding the version. This is useful to build platform-specific
    # wheels (e.g. CPU, TPU) without modifying the source.
    if env_version := os.getenv("APHRODITE_VERSION_OVERRIDE"):
        print(f"Overriding APHRODITE version with {env_version} from APHRODITE_VERSION_OVERRIDE")
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = env_version
        return get_version(write_to="aphrodite/_version.py")

    version = get_version(write_to="aphrodite/_version.py")
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _no_device():
        if envs.APHRODITE_TARGET_DEVICE == "empty":
            version += f"{sep}empty"
    elif _is_cuda():
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
    elif _is_tpu():
        version += f"{sep}tpu"
    elif _is_cpu():
        # Check the local APHRODITE_TARGET_DEVICE (may be set by auto-detect above),
        # not envs.APHRODITE_TARGET_DEVICE, so CPU-only hosts still get `+cpu`.
        if APHRODITE_TARGET_DEVICE == "cpu":
            version += f"{sep}cpu"
    elif _is_metal():
        version += f"{sep}metal"
    elif _is_xpu():
        version += f"{sep}xpu"
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
            if "flashinfer-cubin" in req:
                # Not on PyPI since 0.6.14 (only https://flashinfer.ai/whl), so
                # it cannot be a wheel dependency; flashinfer falls back to
                # fetching cubins at runtime when the package is absent.
                continue
            if "nvidia-cutlass-dsl[cu13]" in req and cuda_major == "12":
                # [cu13] extra is the default; strip it on CUDA 12 builds.
                req = req.replace("nvidia-cutlass-dsl[cu13]", "nvidia-cutlass-dsl")
            if "humming-kernels[cu13]" in req and cuda_major == "12":
                req = req.replace("humming-kernels[cu13]", "humming-kernels[cu12]")
            modified_requirements.append(req)
        requirements = modified_requirements
    elif _is_hip():
        requirements = _read_requirements("rocm.txt")
    elif _is_tpu():
        requirements = _read_requirements("tpu.txt")
    elif _is_cpu():
        requirements = _read_requirements("cpu.txt")
    elif _is_metal():
        requirements = _read_requirements("metal.txt")
    elif _is_xpu():
        requirements = _read_requirements("xpu.txt")
    else:
        raise ValueError("Unsupported platform, please use CUDA, ROCm, Metal, or CPU.")
    return requirements


ext_modules = []

if _is_cuda() or _is_hip():
    ext_modules.append(CMakeExtension(name="aphrodite.cumem_allocator"))
    # Optional since this doesn't get built (produce an .so file). This is just
    # copying the relevant .py files from the source repository.
    ext_modules.append(CMakeExtension(name="aphrodite.triton_kernels", optional=True))

if sys.version_info >= (3, 11):
    ext_modules.append(CMakeExtension(name="aphrodite.spinloop"))
    ext_modules.append(CMakeExtension(name="aphrodite.fs_io_C"))

if _is_hip():
    ext_modules.append(CMakeExtension(name="aphrodite._rocm_C"))

if _is_cuda():
    ext_modules.append(CMakeExtension(name="aphrodite.vllm_flash_attn._vllm_fa2_C"))
    if CUDA_HOME and get_nvcc_cuda_version() >= Version("12.3"):
        # FA3 requires CUDA 12.3 or later
        ext_modules.append(CMakeExtension(name="aphrodite.vllm_flash_attn._vllm_fa3_C"))
    # FA4 CuteDSL - Python-only component for FA4's cute DSL support
    # Optional since this doesn't produce a .so file, just copies Python files
    ext_modules.append(CMakeExtension(name="aphrodite.vllm_flash_attn._vllm_fa4_cutedsl_C", optional=True))
    if CUDA_HOME and get_nvcc_cuda_version() >= Version("12.9"):
        # FlashMLA requires CUDA 12.9 or later
        # Optional since this doesn't get built (produce an .so file) when
        # not targeting a hopper system
        ext_modules.append(CMakeExtension(name="aphrodite._flashmla_C", optional=True))
        ext_modules.append(CMakeExtension(name="aphrodite._flashmla_extension_C", optional=True))
    if CUDA_HOME and get_nvcc_cuda_version() >= Version("12.3"):
        # DeepGEMM requires CUDA 12.3+ (SM90/SM100)
        # Optional since it won't build on unsupported architectures
        ext_modules.append(CMakeExtension(name="aphrodite._deep_gemm_C", optional=True))
        ext_modules.append(CMakeExtension(name="aphrodite._qutlass_C", optional=True))
    # fmha_sm100 is a Python/CuTe-DSL package installed into aphrodite.third_party.
    ext_modules.append(CMakeExtension(name="aphrodite.fmha_sm100", optional=True))
    # tml-fa4 is copied into an isolated aphrodite.third_party package.
    ext_modules.append(CMakeExtension(name="aphrodite.tml_fa4", optional=True))

if _is_cpu():
    import platform

    if platform.machine() in ("x86_64", "AMD64"):
        ext_modules.append(CMakeExtension(name="aphrodite._C"))
        ext_modules.append(CMakeExtension(name="aphrodite._C_AVX512"))
        ext_modules.append(CMakeExtension(name="aphrodite._C_AVX2"))
    else:
        ext_modules.append(CMakeExtension(name="aphrodite._C"))

if _build_custom_ops():
    if _is_metal():
        # MLX/nanobind paged-attention Metal kernel (aphrodite/metal/metal/_paged_ops).
        ext_modules.append(CMakeExtension(name="aphrodite.metal.metal._paged_ops", py_limited_api=False))
    if _is_hip():
        ext_modules.append(CMakeExtension(name="aphrodite._C"))
    if _is_cuda() or _is_hip():
        ext_modules.append(CMakeExtension(name="aphrodite._C_stable_libtorch"))
        ext_modules.append(CMakeExtension(name="aphrodite._moe_C_stable_libtorch"))
        # Fork-only kernels (DRY sampler + EXL3), registered into torch.ops._C.
        ext_modules.append(CMakeExtension(name="aphrodite._C_fork"))

package_data = {
    "aphrodite": [
        "py.typed",
        "libs/*.so*",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/mamba/ops/configs/selective_state_update/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
        "entrypoints/serve/instrumentator/static/*.js",
        "entrypoints/serve/instrumentator/static/*.css",
        "distributed/kv_transfer/kv_connector/v1/hf3fs/utils/*.cpp",
        "third_party/flash_linear_attention/LICENSE",
        # DeepGEMM JIT include headers (vendored via cmake)
        "third_party/deep_gemm/include/**/*.cuh",
        "third_party/deep_gemm/include/**/*.h",
        "third_party/deep_gemm/include/**/*.hpp",
        # fmha_sm100 sparse CuTe-DSL helper kernels (vendored via cmake)
        "third_party/fmha_sm100/csrc/**/*.cu",
        "third_party/fmha_sm100/csrc/**/*.h",
        "third_party/fmha_sm100/csrc/**/*.jinja",
        "third_party/fmha_sm100/csrc/**/*.cu.jinja",
        "third_party/fmha_sm100/cute/**/*.cu",
        "third_party/fmha_sm100/cutlass/include/**/*.h",
        "third_party/fmha_sm100/cutlass/include/**/*.hpp",
        "third_party/fmha_sm100/cutlass/tools/util/include/**/*.h",
        "third_party/fmha_sm100/cutlass/tools/util/include/**/*.hpp",
        # tml-fa4 CuTe-DSL helper kernels (vendored via cmake)
        "third_party/tml_fa4/**/*.py",
    ]
}


def add_aphrodite_package_data(filename: str) -> None:
    aphrodite_files = package_data.setdefault("aphrodite", [])
    if filename not in aphrodite_files:
        aphrodite_files.append(filename)


# If the rust frontend binary is already present in the source tree (e.g.,
# pre-built in a separate Docker build stage), ship it as-is.
if PRECOMPILED_RUST_FRONTEND_PATH.exists():
    add_aphrodite_package_data("aphrodite-rs")
for rust_extension_path in get_precompiled_rust_extension_paths():
    add_aphrodite_package_data(rust_extension_path.name)

if _no_device():
    ext_modules = []

if APHRODITE_USE_PRECOMPILED:
    prebuilt = sorted(
        p.relative_to(ROOT_DIR / "aphrodite")
        for pattern in ("*.so", "vllm_flash_attn/*.so", "third_party/deep_gemm/*.so", "metal/metal/*.so")
        for p in (ROOT_DIR / "aphrodite").glob(pattern)
    )
    if prebuilt:
        logger.info(
            "APHRODITE_USE_PRECOMPILED=1: skipping native builds; reusing %d prebuilt "
            "extension(s) found in the source tree",
            len(prebuilt),
        )
        for rel in prebuilt:
            add_aphrodite_package_data(str(rel))
    else:
        logger.warning(
            "APHRODITE_USE_PRECOMPILED=1 but no prebuilt extensions (*.so) were found "
            "in the source tree — the resulting install will have no native ops. "
            "Run a normal build first if that is not what you want."
        )
    ext_modules = []

if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {"build_ext": cmake_build_ext}

# Rust artifacts, built via setuptools-rust and installed into the package
# directory alongside the Python modules. Imported lazily: setuptools-rust does
# not need to be installed when nothing is being built.
if APHRODITE_USE_PRECOMPILED or should_use_precompiled_rust():
    rust_extensions = []
else:
    rust_build = load_module_from_path("rust_build", os.path.join(ROOT_DIR, "tools", "build_rust.py"))
    rust_extensions = rust_build.rust_extensions(optional=not should_require_rust_frontend())

setup(
    # static metadata should rather go in pyproject.toml
    version=get_aphrodite_version(),
    ext_modules=ext_modules,
    rust_extensions=rust_extensions,
    install_requires=get_requirements(),
    extras_require={
        # AMD Zen CPU optimizations via zentorch
        "zen": ["zentorch==2.11.0.0"],
        "bench": ["pandas", "matplotlib", "seaborn", "datasets", "scipy", "plotly"],
        "tensorizer": ["tensorizer==2.10.1"],
        "fastsafetensors": ["fastsafetensors >= 0.3.2"],
        "instanttensor": ["instanttensor >= 0.1.5"],
        "runai": ["runai-model-streamer[s3,gcs,azure] >= 0.15.7"],
        "audio": [
            "av",
            "scipy",
            "soundfile",
            "soxr",
            "mistral_common[audio]",
        ],  # Required for audio processing
        "video": [],  # Kept for backwards compatibility
        # NVIDIA DeepStream (NVDEC) GPU video-decode backend. Linux x86-64
        # only; also needs system GStreamer + libv4l (see docs).
        "deepstream": ["nvidia-deepstream-videodecode-cu13>=9.0.2"],
        "flashinfer": [],  # Kept for backwards compatibility
        # Optional deps for Helion kernel development
        # NOTE: When updating helion version, also update CI files:
        #   - .buildkite/test_areas/kernels.yaml
        #   - .buildkite/test-amd.yaml
        "helion": ["helion==1.1.0"],
        # Optional deps for gRPC server (aphrodite serve --grpc)
        "grpc": ["smg-grpc-servicer[aphrodite] >= 0.5.2"],
        # Optional deps for OpenTelemetry tracing
        "otel": [
            "opentelemetry-sdk>=1.26.0",
            "opentelemetry-api>=1.26.0",
            "opentelemetry-exporter-otlp>=1.26.0",
            "opentelemetry-semantic-conventions-ai>=0.4.1",
        ],
        # extra quantization plugin
        "extra-quant": ["aphrodite-gguf-plugin>=0.0.2"],
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
