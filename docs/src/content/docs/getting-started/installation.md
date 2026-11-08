---
title: Install Sonar
description: Install Sonar on NVIDIA, AMD, Intel, CPU, Apple, and TPU systems.
---

Sonar supports Linux and macOS. Use WSL 2 to run Sonar on Windows. Native
Windows builds are not supported.

The Python distribution, command, and import are named `sonar`. The `sonar`
project on PyPI is unrelated: bare `pip install sonar` or `uv pip install sonar`
installs that project. Use the installer below or the Sonar wheel index for your
platform. Keep the index arguments and `--index-strategy first-index` when
installing or upgrading with uv.

## Automatic installer

The installer detects the operating system, CPU architecture, and accelerator.
It selects a release or nightly wheel for that platform.

```bash
curl -LsSf https://sonar.dphn.ai/install.sh | sh
```

The installer can create `~/.sonar_venv` with CPython 3.13. You can also select
an existing virtual environment. The installer checks its Python version and
architecture before it changes the environment.

Use explicit options for unattended installation.

```bash
curl -LsSf https://sonar.dphn.ai/install.sh | sh -s -- \
  --yes \
  --channel release \
  --venv ~/.sonar_venv
```

Use `--backend` to override hardware detection. Accepted values are `cuda`,
`rocm`, `xpu`, `metal`, and `cpu`. Run the installer with `--help` to see all
options.

Linux CPU wheels require the `libnuma` runtime library. The installer supports
APT, DNF, YUM, Zypper, and Pacman. It asks before it uses `sudo`. If `sudo` is
not available, it prints the required package-manager command and exits.

## NVIDIA CUDA

CUDA wheels support Linux x86-64 and ARM64 (`aarch64`) with Python 3.11-3.14
and a driver supporting CUDA 13.0 or newer. Supported compute capabilities are
8.0, 8.6, 8.9, 9.0, 10.0, 10.3, and 12.x. ARM64 wheels also support 11.0
(Jetson Thor). Jetson Orin (8.7) and CUDA 12 drivers are not supported.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then
create an environment and install Sonar.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install sonar \
  --extra-index-url https://sonar.dphn.ai/whl/release/cuda/x86_64/simple \
  --index-strategy first-index \
  --torch-backend cu130
```

Use a nightly build when you need the latest commit.

```bash
uv pip install sonar \
  --extra-index-url https://sonar.dphn.ai/whl/nightly/cuda/x86_64/simple \
  --index-strategy first-index \
  --torch-backend cu130
```

On ARM64, replace `cuda/x86_64` with `cuda/aarch64` in either index URL.
The automatic installer selects this index when it detects an NVIDIA GPU on
Linux ARM64. You can also select it with `--backend cuda`.

### Develop with precompiled extensions

Use the per-commit nightly wheel when you change Python files only. Sonar takes
the native extensions from the wheel for the current Git commit and installs
the Python source in editable mode.

```bash
git clone https://github.com/dphnAI/sonar.git
cd sonar
uv venv --python 3.12 --seed
source .venv/bin/activate
SONAR_USE_PRECOMPILED=1 \
  uv pip install --editable . --torch-backend=cu130
```

The commit must have a compatible wheel at
`https://sonar.dphn.ai/commits/<commit>/`. Build the native extensions when you
change `csrc/`, CMake, Rust, or a bundled native dependency.

### Build native extensions incrementally

CUDA source builds require GCC 13 or newer, its matching C++ standard library,
and the development packages for `libelf` and `libdw`. DeepGEMM and DeepSelect
require C++20 `<format>` support. The CUDA Docker image installs these
dependencies for builds and runtime JIT compilation.

Install the development environment first. Then generate a CMake preset from
the active virtual environment.

```bash
python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
```

The generator detects Python, CUDA, Ninja, the available CPU cores, and
`ccache` or `sccache`. It stores the configuration in
`CMakeUserPresets.json`. CMake stores the build state in
`cmake-build-release/`.

After you change native code, run only the build command:

```bash
cmake --build --preset release --target install
```

Ninja recompiles the affected files and reuses the other objects. The
`install` target places the updated extensions in the source tree.

Use `uv pip install --editable .` to create the development environment. Do not
use it as the normal native-code rebuild command. It runs the Python packaging
backend and repeats package build work. A direct CMake build keeps one
configured build tree, exposes incremental build output, and gives more
consistent rebuild times.

## Docker

Build the serving image from the current source tree.

```bash
git clone https://github.com/dphnAI/sonar.git
cd sonar
docker build -f docker/Dockerfile -t sonar .
docker run --rm --gpus all --ipc=host \
  -p 2242:2242 \
  sonar \
  --model Qwen/Qwen3-0.6B
```

## AMD ROCm

ROCm support is source-based. Use the ROCm requirements and select the ROCm
target.

```bash
git clone https://github.com/dphnAI/sonar.git
cd sonar
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements/build.txt -r requirements/rocm.txt
SONAR_TARGET_DEVICE=rocm uv pip install . --no-build-isolation
```

Install a compatible ROCm and PyTorch build before this procedure. The supported
GPU architectures follow the versions pinned in `requirements/rocm.txt`.

## Intel XPU

Install the Intel GPU driver and oneAPI. Then build with the XPU requirements.

```bash
git clone https://github.com/dphnAI/sonar.git
cd sonar
source /opt/intel/oneapi/setvars.sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements/build.txt -r requirements/xpu.txt
SONAR_TARGET_DEVICE=xpu uv pip install . --no-build-isolation
```

## CPU

The CPU backend supports Linux x86-64 and Arm64. Build it from source.

```bash
git clone https://github.com/dphnAI/sonar.git
cd sonar
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements/build/cpu.txt -r requirements/cpu.txt
SONAR_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

Set `SONAR_CPU_KVCACHE_SPACE` to the KV cache size in GiB. Start with a
conservative value if other processes use the same host.

## Apple silicon

The Metal backend requires an Apple silicon Mac.

```bash
git clone https://github.com/dphnAI/sonar.git
cd sonar
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements/metal.txt
SONAR_TARGET_DEVICE=metal uv pip install . --no-build-isolation
```

Metal supports hybrid attention models, but it cannot reuse their recurrent
state across requests. Sonar disables prefix caching for these models and logs
a warning. Prefix caching remains available for pure-attention models.

## Google TPU

TPU support requires a TPU VM and a compatible PyTorch XLA installation.

```bash
git clone https://github.com/dphnAI/sonar.git
cd sonar
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements/tpu.txt
SONAR_TARGET_DEVICE=tpu uv pip install . --no-build-isolation
```

Use the PyTorch XLA installation procedure for your TPU image before you build
Sonar.
