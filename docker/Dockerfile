# The Sonar Dockerfile is used to construct Sonar image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/contributing/dockerfile/dockerfile.md and
# docs/assets/contributing/dockerfile-stages-dependency.png

# =============================================================================
# VERSION MANAGEMENT
# =============================================================================
# ARG defaults in this Dockerfile are the source of truth for pinned versions.
# docker/versions.json is auto-generated for use with docker buildx bake.
#
# When updating versions:
# 1. Edit the ARG defaults below
# 2. Run: python tools/generate_versions_json.py
#
# To query versions programmatically:
#   jq -r '.variable.CUDA_VERSION.default' docker/versions.json
#
# To build with bake:
#   docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json
# =============================================================================

ARG CUDA_VERSION=13.2.1
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION=22.04
ARG INSTALL_RUBIN_PRERELEASE=false
ARG TRITON_INSTALL_FROM_SOURCE_REPO=
ARG TRITON_INSTALL_FROM_SOURCE_REVISION=
# DeepEPv2 requires NCCL >= 2.30.4 (GIN backend).
# This version is used for CUDA 12+ builds to keep the DeepEP build and
# runtime NCCL versions compatible.
# DeepEPv2 GDA over EFA requires NCCL 2.31.x or later.
ARG NCCL_VERSION=2.31.2
# AWS EFA userspace (libfabric, aws-ofi-nccl), installed with the KV connectors.
ARG EFA_INSTALLER_VERSION=1.51.0
ARG EFA_INSTALLER_SHA256=83993044fa4df3dc2f8b5c2f7051d61b9689441be023a2c92ea51d3f53f2465c

# By parameterizing the base images, we allow third-party to use their own
# base images. One use case is hermetic builds with base images stored in
# private registries that use a different repository naming conventions.
#
# Example:
# docker build --build-arg BUILD_BASE_IMAGE=registry.acme.org/mirror/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Important: We build with an old version of Ubuntu to maintain broad
# compatibility with other Linux OSes. The main reason for this is that the
# glibc version is baked into the distro, and binaries built with one glibc
# version are not backwards compatible with OSes that use an earlier version.
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
# Using cuda base image with minimal dependencies necessary for JIT compilation (FlashInfer, DeepGEMM, EP kernels)
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# By parameterizing the Deadsnakes repository URL, we allow third-party to use
# their own mirror. When doing so, we don't benefit from the transparent
# installation of the GPG key of the PPA, as done by add-apt-repository, so we
# also need a URL for the GPG key.
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL

# The PyPA get-pip.py script is a self contained script+zip file, that provides
# both the installer script and the pip base85-encoded zip archive. This allows
# bootstrapping pip in environment where a distribution package does not exist.
#
# By parameterizing the URL for get-pip.py installation script, we allow
# third-party to use their own copy of the script stored in a private mirror.
# We set the default value to the PyPA owned get-pip.py script.
#
# Reference: https://pip.pypa.io/en/stable/installation/#get-pip-py
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

# PIP supports fetching the packages from custom indexes, allowing third-party
# to host the packages in private mirrors. The PIP_INDEX_URL and
# PIP_EXTRA_INDEX_URL are standard PIP environment variables to override the
# default indexes. By letting them empty by default, PIP will use its default
# indexes if the build process doesn't override the indexes.
#
# Uv uses different variables. We set them by default to the same values as
# PIP, but they can be overridden.
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

# PyTorch provides its own indexes for standard and nightly builds
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl

# PIP supports multiple authentication schemes, including keyring
# By parameterizing the PIP_KEYRING_PROVIDER variable and setting it to
# disabled by default, we allow third-party to use keyring authentication for
# their private Python indexes, while not changing the default behavior which
# is no authentication.
#
# Reference: https://pip.pypa.io/en/stable/topics/authentication/#keyring-support
ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

# Flag enables built-in KV-connector dependency libs into docker images
ARG INSTALL_KV_CONNECTORS=false

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM ${BUILD_BASE_IMAGE} AS base

ARG TARGETPLATFORM
ARG INSTALL_RUBIN_PRERELEASE
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG USE_SCCACHE
ARG SCCACHE_DOWNLOAD_URL
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=aphrodite-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_INSTALL_DIR=/opt/uv/bin

# Install system dependencies including build tools
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache \
        software-properties-common \
        git \
        curl \
        sudo \
        python3-pip \
        libibverbs-dev \
        libelf-dev \
        libdw-dev \
    # DeepGEMM and DeepSelect require GCC 13's C++20 <format> support.
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5" \
        | gpg --dearmor -o /usr/share/keyrings/ubuntu-toolchain-r.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ubuntu-toolchain-r.gpg] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
        > /etc/apt/sources.list.d/ubuntu-toolchain-r.list \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends gcc-13 g++-13 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    # Install python dev headers if available (needed for cmake FindPython on Ubuntu 24.04
    # which ships cmake 3.28 and requires Development.SABIModule; silently skipped on
    # Ubuntu 20.04/22.04 where python3.x-dev is not available without a PPA)
    && (apt-get install -y --no-install-recommends python${PYTHON_VERSION}-dev 2>/dev/null || true) \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /opt/uv/bin/uv venv --seed /opt/venv --python ${PYTHON_VERSION} \
    && rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip \
    && ln -s /opt/venv/bin/python3 /usr/bin/python3 \
    && ln -s /opt/venv/bin/python3-config /usr/bin/python3-config \
    && ln -s /opt/venv/bin/pip /usr/bin/pip \
    && python3 --version && python3 -m pip --version

# Install sccache once in base so Rust and CMake/CUDA builds share the same
# binary and remote cache configuration.
RUN if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && case "${TARGETPLATFORM}" in \
          linux/arm64) SCCACHE_ARCH="aarch64" ;; \
          linux/amd64) SCCACHE_ARCH="x86_64" ;; \
          *) echo "Unsupported TARGETPLATFORM for sccache: ${TARGETPLATFORM}" >&2; exit 1 ;; \
        esac \
        && SCCACHE_VERSION="v0.8.1" \
        && SCCACHE_DL_URL="${SCCACHE_DOWNLOAD_URL:-https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl.tar.gz}" \
        && curl -L -o /tmp/sccache.tar.gz "${SCCACHE_DL_URL}" \
        && tar -xzf /tmp/sccache.tar.gz -C /tmp \
        && mv /tmp/sccache-${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl/sccache /usr/bin/sccache \
        && chmod +x /usr/bin/sccache \
        && rm -rf /tmp/sccache.tar.gz /tmp/sccache-${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl; \
    fi

ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET_NAME}}
ENV SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION_NAME}}
ENV SCCACHE_S3_NO_CREDENTIALS=${USE_SCCACHE:+${SCCACHE_S3_NO_CREDENTIALS}}
ENV SCCACHE_IDLE_TIMEOUT=${USE_SCCACHE:+0}

# Activate virtual environment and add uv to PATH
ENV PATH="/opt/venv/bin:/opt/uv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Environment for uv
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Verify GCC version
RUN gcc --version

# Enable CUDA forward compatibility by setting '-e SONAR_ENABLE_CUDA_COMPATIBILITY=1'
# Only needed for datacenter/professional GPUs with older drivers.
# See: https://docs.nvidia.com/deploy/cuda-compatibility/
ENV SONAR_ENABLE_CUDA_COMPATIBILITY=0

# ============================================================
# SLOW-CHANGING DEPENDENCIES BELOW
# These are the expensive layers that we want to cache
# ============================================================

# Install PyTorch and core CUDA dependencies
# This is ~2GB and rarely changes
ARG PYTORCH_CUDA_INDEX_BASE_URL

WORKDIR /workspace

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install build and runtime dependencies, including PyTorch
# Check whether to install torch nightly instead of release for this build
COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
COPY tools/use_existing_torch.py tools/use_existing_torch.py
COPY pyproject.toml pyproject.toml
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=requirements/rubin-prerelease.txt,target=/tmp/sonar-requirements/rubin-prerelease.txt,ro \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ] && \
        [ ! -f /tmp/sonar-requirements/rubin-prerelease.txt ]; then \
        echo "INSTALL_RUBIN_PRERELEASE=true requires requirements/rubin-prerelease.txt" >&2; \
        exit 1; \
    fi \
    && if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' requirements/cuda.txt; \
        sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' requirements/cuda.txt; \
    fi \
    && if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        echo "Installing Rubin prerelease requirements..." \
        && /opt/venv/bin/python3 tools/use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/sonar-requirements/rubin-prerelease.txt \
        && uv pip install --python /opt/venv/bin/python3 \
            -r requirements/cuda.txt \
        && uv pip install --python /opt/venv/bin/python3 \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/sonar-requirements/rubin-prerelease.txt; \
    elif [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing torch nightly..." \
        && uv pip install --python /opt/venv/bin/python3 torch torchaudio torchvision --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        && echo "Installing other requirements..." \
        && /opt/venv/bin/python3 tools/use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        # Keep the CUDA-specific PyTorch index first. Using it as an extra
        # index with unsafe-best-match can select a higher, incompatible
        # torch build from the base image (for example cu130 in a cu129 image).
        uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match; \
    fi

# Track PyTorch lib versions used during build and match in downstream instances.
# We do this for both nightly and release so we can strip dependencies/*.txt as needed.
# Otherwise library dependencies can upgrade/downgrade torch incorrectly.
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip freeze | grep -i "^torch=\|^torchvision=\|^torchaudio=" > torch_lib_versions.txt \
    && TORCH_LIB_VERSIONS=$(cat torch_lib_versions.txt | xargs) \
    && echo "Installed torch libs: ${TORCH_LIB_VERSIONS}"

# CUDA arch list used by torch
# Explicitly set the list to avoid issues with torch 2.2
# See https://github.com/pytorch/pytorch/pull/123243
# From versions.json: .torch.cuda_arch_list
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BUILD BASE IMAGE ####################

#################### HYBRID TEST WHEELS IMAGE ####################
# mamba-ssm and causal-conv1d for the hybrid-model tests, built from pinned
# commits against base's torch. It rebuilds with base, in parallel with the
# longer csrc-build, so it adds no time to the image build. No arm64 or Rubin
# job runs these tests, so those builds leave /wheels empty. A failed build
# fails the image, except on torch nightly: its torch changes every run, so it
# warns, and only the hybrid jobs fail (in tools/check_hybrid_test_deps.py) instead
# of the whole nightly run.
FROM base AS hybrid-test-wheels
ARG TARGETPLATFORM
ARG INSTALL_RUBIN_PRERELEASE
ARG PYTORCH_NIGHTLY
ARG max_jobs=2
WORKDIR /workspace
COPY requirements/build/hybrid-test-deps.txt requirements/build/hybrid-test-deps.txt
COPY tools/build_hybrid_test_wheels.sh tools/build_hybrid_test_wheels.sh
RUN --mount=type=cache,target=/opt/uv/cache \
    mkdir -p /wheels \
    && if [ "$TARGETPLATFORM" = "linux/arm64" ] || [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then exit 0; fi \
    && uv pip install --python /opt/venv/bin/python3 --require-hashes --reinstall \
        -r requirements/build/hybrid-test-deps.txt \
    && { MAX_JOBS=${max_jobs} bash tools/build_hybrid_test_wheels.sh /wheels \
         || { [ "${PYTORCH_NIGHTLY}" = "1" ] \
              && echo "WARNING: mamba-ssm/causal-conv1d failed to build against torch nightly" \
              && rm -rf /wheels/*; }; }
#################### HYBRID TEST WHEELS IMAGE ####################

#################### RUST BUILD IMAGE ####################
# Build the Rust frontend separately so the CUDA wheel build does not need a
# Rust toolchain and can reuse Cargo artifacts across wheel commits.
FROM scratch AS rust-input
COPY requirements/build/rust.txt /rust-input/sonar/requirements/build/rust.txt
COPY rust/Cargo.lock rust/Cargo.toml /rust-input/sonar/rust/
COPY rust/proto /rust-input/sonar/rust/proto
COPY rust/src /rust-input/sonar/rust/src
COPY tools/build_rust.py /rust-input/sonar/tools/build_rust.py

FROM base AS rust-build
ARG USE_SCCACHE
ARG SCCACHE_ENDPOINT

ENV CARGO_BUILD_JOBS=4
ENV CARGO_NET_RETRY=10
ENV RUSTUP_MAX_RETRIES=10
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH=${CARGO_HOME}/bin:${PATH}
ENV SCCACHE_SERVER_PORT=4227

WORKDIR /workspace

COPY --from=rust-input \
    /rust-input/sonar/requirements/build/rust.txt \
    requirements/build/rust.txt

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        make \
        unzip \
        protobuf-compiler \
        libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustc --version \
    && cargo --version

RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install \
      --python /opt/venv/bin/python3 \
      -r requirements/build/rust.txt

COPY --from=rust-input /rust-input/sonar/rust rust
COPY --from=rust-input \
    /rust-input/sonar/tools/build_rust.py \
    tools/build_rust.py

RUN --mount=type=cache,id=sonar-cuda-cargo-registry,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=sonar-cuda-cargo-git,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,id=sonar-cuda-cargo-target,target=/workspace/rust/target,sharing=locked \
    --mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false \
    if [ "$USE_SCCACHE" = "1" ]; then \
        if [ -n "${SCCACHE_ENDPOINT}" ]; then export SCCACHE_ENDPOINT="${SCCACHE_ENDPOINT}"; fi; \
        export RUSTC_WRAPPER=sccache; \
        sccache --show-stats; \
    fi \
    && python3 tools/build_rust.py --release \
    && if [ "$USE_SCCACHE" = "1" ]; then \
        sccache --show-stats; \
    fi

#################### RUST BUILD IMAGE ####################


#################### EXTENSIONS BUILD IMAGE ####################
# Build DeepEP wheels
FROM base AS extensions-build
ARG CUDA_VERSION
ARG TARGETARCH
ARG TRITON_INSTALL_FROM_SOURCE_REPO
ARG TRITON_INSTALL_FROM_SOURCE_REVISION
ARG max_jobs

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# Build DeepEP wheels
COPY tools/ep_kernels/install_python_libraries.sh /tmp/install_python_libraries.sh
COPY tools/ep_kernels/patches /tmp/patches
# Defaults moved here from tools/ep_kernels/install_python_libraries.sh for centralized version management
ARG DEEPEP_COMMIT_HASH=a56d6156febcd9976e55adc85b5155bfac9f28f8
ARG NVSHMEM_VER
ARG NCCL_VERSION
RUN --mount=type=cache,target=/opt/uv/cache \
    mkdir -p /tmp/ep_kernels_workspace/dist && \
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1) && \
    if [ -n "$NCCL_VERSION" ]; then \
        echo "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}" \
            > /tmp/nccl-override.txt && \
        export UV_OVERRIDE=/tmp/nccl-override.txt; \
    fi && \
    DEEPEP_ARCH_LIST='9.0a 10.0a' && \
    if [ "$(printf '%s\n' 12.9 "$CUDA_VERSION" | sort -V | head -n1)" = 12.9 ]; then \
        DEEPEP_ARCH_LIST="$DEEPEP_ARCH_LIST 10.3a"; \
    fi && \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        DEEPEP_ARCH_LIST="$DEEPEP_ARCH_LIST 10.7a"; \
    fi && \
    export TORCH_CUDA_ARCH_LIST="$DEEPEP_ARCH_LIST" && \
    /tmp/install_python_libraries.sh \
        --workspace /tmp/ep_kernels_workspace \
        --mode wheel \
        ${DEEPEP_COMMIT_HASH:+--deepep-ref "$DEEPEP_COMMIT_HASH"} \
        ${NVSHMEM_VER:+--nvshmem-ver "$NVSHMEM_VER"} && \
    find /tmp/ep_kernels_workspace/nvshmem -name '*.a' -delete

# Build Triton from source when a repository is configured. Triton 3.8 pins
# ptxas-blackwell 13.3, which cannot assemble SM107. Rubin CUDA 13.4 and 13.5
# builds use their SM107-capable system ptxas during the wheel build and at
# runtime.
COPY tools/build_triton_from_source.sh /tmp/build_triton_from_source.sh
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=cache,id=triton-source-${TARGETARCH},target=/root/.triton,sharing=locked \
    --mount=type=cache,id=triton-source-ccache-${TARGETARCH},target=/root/.cache/ccache,sharing=locked \
    set -eux; \
    TRITON_WORKSPACE=/tmp/triton_from_source_workspace; \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ] \
        && [ -z "${TRITON_INSTALL_FROM_SOURCE_REPO}" ]; then \
        echo "Rubin prerelease builds require TRITON_INSTALL_FROM_SOURCE_REPO" >&2; \
        exit 1; \
    elif [ -z "${TRITON_INSTALL_FROM_SOURCE_REPO}" ] \
        && [ -n "${TRITON_INSTALL_FROM_SOURCE_REVISION}" ]; then \
        echo "TRITON_INSTALL_FROM_SOURCE_REVISION requires TRITON_INSTALL_FROM_SOURCE_REPO" >&2; \
        exit 1; \
    elif [ -n "${TRITON_INSTALL_FROM_SOURCE_REPO}" ]; then \
        apt-get update -y; \
        apt-get install -y --no-install-recommends zlib1g-dev; \
        rm -rf /var/lib/apt/lists/*; \
        CUDA_MAJOR_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f1,2); \
        if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
            case "${CUDA_MAJOR_MINOR}" in \
                13.4|13.5) \
                    export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
                    ;; \
                *) \
                    echo "Unsupported Rubin CUDA version: ${CUDA_MAJOR_MINOR}" >&2; \
                    exit 1 \
                    ;; \
            esac; \
        fi; \
        TRITON_WORKSPACE="${TRITON_WORKSPACE}" \
        MAX_JOBS="${max_jobs:-}" \
            bash /tmp/build_triton_from_source.sh; \
    else \
        mkdir -p "${TRITON_WORKSPACE}/dist"; \
        printf '%s\n' disabled > "${TRITON_WORKSPACE}/dist/build-state"; \
        (cd "${TRITON_WORKSPACE}/dist" \
            && sha256sum build-state > wheel.sha256); \
    fi
#################### EXTENSIONS BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install build dependencies
COPY requirements/build/cuda.txt requirements/build/cuda.txt
COPY tools/use_existing_torch.py tools/use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        echo "Installing build requirements with Rubin prerelease PyTorch..." \
        && python3 tools/use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt; \
    elif [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing build requirements without torch..." \
        && python3 tools/use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        && echo "Installing torch nightly..." \
        && uv pip install --python /opt/venv/bin/python3 $(cat torch_lib_versions.txt | grep -i "^torch=" | xargs) --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing build requirements..." \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match; \
    fi

WORKDIR /workspace

COPY . .

# setup.py packages these separately built Rust artifacts as-is.
RUN rm -f sonar/sonar-rs sonar/_rust_*.so
COPY --from=rust-build /workspace/sonar/sonar-rs sonar/sonar-rs
COPY --from=rust-build /workspace/sonar/_rust_*.so sonar/

ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

ARG sonar_target_device="cuda"
ARG SONAR_VERSION_OVERRIDE=""
ARG max_jobs=2
ARG nvcc_threads=8
ARG python_abi_tag=cp38
ARG USE_SCCACHE
ARG SCCACHE_ENDPOINT
ENV SONAR_TARGET_DEVICE=${sonar_target_device}
ENV SONAR_VERSION_OVERRIDE=${SONAR_VERSION_OVERRIDE}
ENV MAX_JOBS=${max_jobs}
ENV NVCC_THREADS=${nvcc_threads}
ENV CMAKE_BUILD_TYPE=Release
ENV SONAR_USE_PRECOMPILED_RUST=1

# Skip adding +precompiled suffix to version (preserves git-derived version)
ENV SONAR_SKIP_PRECOMPILED_VERSION_SUFFIX=1

# Use existing torch for nightly and Rubin prerelease builds
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ] || \
        [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        python3 tools/use_existing_torch.py --prefix; \
    fi

# Build the Sonar wheel
RUN --mount=type=cache,id=sonar-wheel-uv,target=/opt/uv/cache \
    --mount=type=cache,id=sonar-wheel-ccache-v2,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,id=sonar-wheel-deps-v2,target=/workspace/.deps,sharing=locked \
    --mount=type=bind,source=.git,target=.git \
    --mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false \
    if [ "$USE_SCCACHE" = "1" ]; then \
        if [ -n "${SCCACHE_ENDPOINT}" ]; then export SCCACHE_ENDPOINT="${SCCACHE_ENDPOINT}"; fi; \
        export RUSTC_WRAPPER=sccache; \
        sccache --show-stats; \
    fi \
    && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=${python_abi_tag} \
    && if [ "$USE_SCCACHE" = "1" ]; then \
        sccache --show-stats; \
    fi

RUN python3 .github/scripts/verify_cuda_wheel_artifacts.py \
    "$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)"

#################### WHEEL BUILD IMAGE ####################

#################### DEV IMAGE ####################
FROM base AS dev

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Install libnuma-dev, required by fastsafetensors (fixes #20384)
RUN apt-get update && apt-get install -y --no-install-recommends libnuma-dev && rm -rf /var/lib/apt/lists/*


# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install development dependencies
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/dev.txt requirements/dev.txt
# uv reads `[tool.uv] exclude-dependencies` from here when recompiling below.
COPY pyproject.toml pyproject.toml
COPY tools/use_existing_torch.py tools/use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing dev requirements plus torch nightly..." \
        && python3 tools/use_existing_torch.py --prefix \
        && cat torch_lib_versions.txt >> requirements/test/cuda.in \
        && uv pip compile --generate-hashes requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        && uv pip install --python /opt/venv/bin/python3 $(cat torch_lib_versions.txt | xargs) --pre \
        -r requirements/dev.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing dev requirements..." \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/dev.txt \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match; \
    fi

#################### DEV IMAGE ####################
#################### SONAR INSTALLATION IMAGE ####################
# image with Sonar installed
FROM ${FINAL_BASE_IMAGE} AS sonar-base

ARG INSTALL_RUBIN_PRERELEASE
ARG TARGETPLATFORM
ARG max_jobs=2
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG NCCL_VERSION
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /sonar-workspace


# Python version string for paths (e.g., "312" for 3.12)
RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and system dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
        curl \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5" \
        | gpg --dearmor -o /usr/share/keyrings/ubuntu-toolchain-r.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ubuntu-toolchain-r.gpg] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
        > /etc/apt/sources.list.d/ubuntu-toolchain-r.list \
    && if [ ! -z ${DEADSNAKES_MIRROR_URL} ] ; then \
        if [ ! -z "${DEADSNAKES_GPGKEY_URL}" ] ; then \
            mkdir -p -m 0755 /etc/apt/keyrings ; \
            curl -L ${DEADSNAKES_GPGKEY_URL} | gpg --dearmor > /etc/apt/keyrings/deadsnakes.gpg ; \
            sudo chmod 644 /etc/apt/keyrings/deadsnakes.gpg ; \
            echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] ${DEADSNAKES_MIRROR_URL} $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes.list ; \
        fi ; \
    else \
        for i in 1 2 3; do \
            add-apt-repository -y ppa:deadsnakes/ppa && break || \
            { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
        done ; \
    fi \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        libibverbs-dev \
        libelf-dev \
        libdw-dev \
        gcc-13 \
        g++-13 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install CUDA development tools for runtime JIT compilation
# (FlashInfer, DeepGEMM, EP kernels all require compilation at runtime)
RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        # FlashInfer's runtime JIT-compiled kernels (e.g. trtllm_fmha_kernel_launcher.cu)
        # #include <nvrtc.h>, which only the -dev package ships.
        cuda-nvrtc-dev-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-${CUDA_VERSION_DASH} \
        # Required by fastsafetensors (fixes #20384)
        libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

# Ensures nvcc/ptxas (installed above, needed at runtime for FlashInfer/
# DeepGEMM/EP kernel JIT compilation) are reachable on PATH. Harmless no-op
# if already set by FINAL_BASE_IMAGE.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install the initialized-snapshot runtime on the architecture supported by
# `sonar snapshot`. CRIU and its CUDA plugin are built from the same pinned
# source. The `cuda-checkpoint`-compatible helper is built from NVIDIA Dynamo's
# Apache-2.0 source against the public CUDA Driver checkpoint APIs instead of
# redistributing NVIDIA's proprietary cuda-checkpoint binary. The helper uses
# checkpoint APIs (CUcheckpointGpuPair) that first appear in the CUDA 13
# headers, so CUDA 12.x images skip the snapshot runtime.
COPY tools/install_snapshot_runtime.sh /usr/local/bin/install_snapshot_runtime.sh
RUN if [ "${TARGETPLATFORM}" = "linux/amd64" ] \
        && [ "${CUDA_VERSION%%.*}" -ge 13 ]; then \
        MAX_JOBS="${max_jobs}" /usr/local/bin/install_snapshot_runtime.sh; \
    else \
        echo "Initialized snapshots are not installed for ${TARGETPLATFORM} with CUDA ${CUDA_VERSION}"; \
    fi

ENV CRIU_CUDA_PLUGIN_DIR=/usr/local/lib/criu
# Install uv for faster pip installs
RUN python3 -m pip install uv

# Environment for uv
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_COMPILE_BYTECODE=1
RUN mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" \
    && chgrp -R 0 /opt/uv \
    && chmod -R g+rwX,a+rX /opt/uv

# DeepEPv2 GIN requires NCCL >= 2.30.4 at both compile and runtime. torch pins
# an older version as a transitive dep; this override forces uv to use our
# pinned version whenever nvidia-nccl-cu* is resolved.
RUN CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1) && \
    if [ -n "$NCCL_VERSION" ]; then \
        echo "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}" \
            > /etc/uv-overrides.txt; \
    else \
        touch /etc/uv-overrides.txt; \
    fi
ENV UV_OVERRIDE=/etc/uv-overrides.txt

# Enable CUDA forward compatibility by setting '-e SONAR_ENABLE_CUDA_COMPATIBILITY=1'
# Only needed for datacenter/professional GPUs with older drivers.
# See: https://docs.nvidia.com/deploy/cuda-compatibility/
ENV SONAR_ENABLE_CUDA_COMPATIBILITY=0
# The final image's compat libraries, which can be newer than torch's CUDA
# (Rubin: CUDA 13.5 image with cu134 torch).
ENV SONAR_CUDA_COMPATIBILITY_PATH=/usr/local/cuda/compat

# ============================================================
# SLOW-CHANGING DEPENDENCIES BELOW
# These are the expensive layers that we want to cache
# ============================================================

# Install PyTorch and core CUDA dependencies
# This is ~2GB and rarely changes
ARG PYTORCH_CUDA_INDEX_BASE_URL
COPY requirements/common.txt /tmp/common.txt
COPY requirements/cuda.txt /tmp/requirements-cuda.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=requirements/rubin-prerelease.txt,target=/tmp/sonar-requirements/rubin-prerelease.txt,ro \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ] && \
        [ ! -f /tmp/sonar-requirements/rubin-prerelease.txt ]; then \
        echo "INSTALL_RUBIN_PRERELEASE=true requires requirements/rubin-prerelease.txt" >&2; \
        exit 1; \
    fi && \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' /tmp/requirements-cuda.txt; \
        sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' /tmp/requirements-cuda.txt; \
    fi && \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        uv pip install --system \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/sonar-requirements/rubin-prerelease.txt \
        && sed -Ei '/^[[:space:]]*(torch|torchvision|torchaudio)[[:space:]]*==/Id' \
            /tmp/requirements-cuda.txt \
        && uv pip install --system -r /tmp/requirements-cuda.txt \
        && uv pip install --system \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/sonar-requirements/rubin-prerelease.txt; \
    else \
        uv pip install --system \
            -r /tmp/requirements-cuda.txt \
            --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            --extra-index-url https://pypi.org/simple \
            --index-strategy unsafe-first-match; \
    fi && \
    rm /tmp/requirements-cuda.txt /tmp/common.txt

# Install FlashInfer's precompiled kernels for all GPU architectures (the default).
# `sonar download-kernels` wraps this command but Sonar is only installed in later
# stages, and running it there would rebuild this large layer for every wheel.
RUN --mount=type=cache,target=/opt/pip/cache \
    PIP_CACHE_DIR=/opt/pip/cache flashinfer download-kernels

# ============================================================
# OPENAI API SERVER DEPENDENCIES
# Pre-install these to avoid reinstalling on every Sonar wheel rebuild
# ============================================================

# Install gdrcopy (saves ~6s per build)
# TODO (huydhn): There is no prebuilt gdrcopy package on 12.9 at the moment
ARG GDRCOPY_CUDA_VERSION=12.8
ARG GDRCOPY_OS_VERSION=Ubuntu22_04
ARG TARGETPLATFORM
COPY tools/install_gdrcopy.sh /tmp/install_gdrcopy.sh
RUN set -eux; \
    case "${TARGETPLATFORM}" in \
      linux/arm64) UUARCH="aarch64" ;; \
      linux/amd64) UUARCH="x64" ;; \
      *) echo "Unsupported TARGETPLATFORM: ${TARGETPLATFORM}" >&2; exit 1 ;; \
    esac; \
    /tmp/install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "${GDRCOPY_CUDA_VERSION}" "${UUARCH}" && \
    rm /tmp/install_gdrcopy.sh

# Install sonar-openai dependencies (saves rebuild time)
# These are stable packages that don't depend on Sonar itself
# From versions.json: .bitsandbytes.x86_64, .bitsandbytes.arm64
# From versions.json: .openai_server_extras.timm, .openai_server_extras.runai_model_streamer,
# .openai_server_extras.gguf (NVFP4 GGUF checkpoints, e.g. RecursiveSI/flint-nvfp4-gguf)
ARG BITSANDBYTES_VERSION_X86=0.46.1
ARG BITSANDBYTES_VERSION_ARM64=0.42.0
ARG TIMM_VERSION=">=1.0.17"
ARG RUNAI_MODEL_STREAMER_VERSION=">=0.17.1"
ARG GGUF_VERSION=">=0.19.0"
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION_ARM64}"; \
    else \
        BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION_X86}"; \
    fi; \
    uv pip install --system accelerate 'modelscope<1.38' \
        "bitsandbytes>=${BITSANDBYTES_VERSION}" "timm${TIMM_VERSION}" "runai-model-streamer[s3,gcs,azure]${RUNAI_MODEL_STREAMER_VERSION}" \
        "gguf${GGUF_VERSION}"

# ============================================================
# SONAR INSTALLATION (depends on build stage)
# ============================================================

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install Sonar wheel first, so that torch etc will be installed.
# Check whether to install torch nightly instead of release for this build.
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/sonar-workspace/dist \
    --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing torch nightly..." \
        && uv pip install --system $(cat torch_lib_versions.txt | xargs) --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        && echo "Installing Sonar..." \
        && uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing Sonar..." \
        && uv pip install --system dist/*.whl --verbose \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match; \
    fi

# Expose the Rust frontend bundled in the wheel on the container's PATH.
RUN ln -s "$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/sonar/sonar-rs" /usr/local/bin/sonar-rs \
    && sonar-rs --version

RUN --mount=type=cache,target=/opt/uv/cache \
. /etc/environment && \
uv pip list

# Pytorch now installs NVSHMEM, setting LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install EP kernels directly from their build stage; wheel-only builds do not need them.
RUN --mount=type=bind,from=extensions-build,src=/tmp/ep_kernels_workspace/dist,target=/sonar-workspace/ep_kernels/dist \
    --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system ep_kernels/dist/*.whl --verbose \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match

# CUDA image changed from /usr/local/nvidia to /usr/local/cuda in 12.8 but will
# return to /usr/local/nvidia in 13.0 to allow container providers to mount drivers
# consistently from the host (see https://github.com/vllm-project/vllm/issues/18859).
# Until then, add /usr/local/nvidia/lib64 before the image cuda path to allow override.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Copy examples and benchmarks at the end to minimize cache invalidation
COPY examples examples
COPY benchmarks benchmarks
COPY ./sonar/collect_env.py .
#################### SONAR INSTALLATION IMAGE ####################
#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses Sonar installed by `pip`
FROM sonar-base AS test

ADD . /sonar-workspace/

ARG PYTHON_VERSION

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN apt-get update -y \
    && apt-get install -y git

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install development dependencies (for testing)
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/dev.txt requirements/dev.txt
# uv reads `[tool.uv] exclude-dependencies` from here when recompiling below.
COPY pyproject.toml pyproject.toml
COPY tools/use_existing_torch.py tools/use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -ge 12 ]; then \
        if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
            echo "Installing dev requirements plus torch nightly..." \
            && python3 tools/use_existing_torch.py --prefix \
            && cat torch_lib_versions.txt >> requirements/test/cuda.in \
            && uv pip compile --generate-hashes requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            && uv pip install --system $(cat torch_lib_versions.txt | xargs) --pre \
            -r requirements/dev.txt \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
        else \
            echo "Installing dev requirements..." \
            && uv pip install --system -r requirements/dev.txt \
            --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            --extra-index-url https://pypi.org/simple \
            --index-strategy unsafe-first-match; \
        fi \
    fi

# KV connectors for the P/D and connector suites, so CI jobs never install them
# at runtime. Lock entries the image lacks are installed hash-verified, and the
# install must not change any preinstalled package. The test stage checks the
# finished environment against the lock. Targets the stable x86_64 CUDA 13 image.
COPY requirements/test/kv_connectors_cuda.txt requirements/test/kv_connectors_cuda.txt
COPY tools/kv_connector_lock.py tools/kv_connector_lock.py
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${CUDA_VERSION%%.*}" = "13" ] && [ "${PYTORCH_NIGHTLY}" != "1" ] \
        && [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
        uv pip freeze --system | sort > /tmp/kv-before.txt \
        && python3 tools/kv_connector_lock.py plan \
            requirements/test/kv_connectors_cuda.txt /tmp/kv-new.txt \
        # The NCCL UV_OVERRIDE would enter --require-hashes as an unhashed requirement.
        && env -u UV_OVERRIDE uv pip install --system --require-hashes --no-deps \
            -r /tmp/kv-new.txt \
        && uv pip freeze --system | sort > /tmp/kv-after.txt \
        && if comm -23 /tmp/kv-before.txt /tmp/kv-after.txt | grep .; then \
            echo "kv_connectors_cuda.txt changed the packages above; recompile it" >&2; \
            exit 1; \
        fi \
        && rm -f /tmp/kv-before.txt /tmp/kv-after.txt /tmp/kv-new.txt; \
    fi

# Prebuilt hybrid-model test extensions. --no-deps --offline: their dependencies
# come only from the hash-pinned test requirements installed above.
RUN --mount=type=bind,from=hybrid-test-wheels,source=/wheels,target=/tmp/hybrid-test-wheels \
    if ls /tmp/hybrid-test-wheels/*.whl >/dev/null 2>&1; then \
        uv pip install --system --no-deps --offline /tmp/hybrid-test-wheels/*.whl \
        && cp /tmp/hybrid-test-wheels/commits.txt /opt/hybrid-test-deps-commits.txt; \
    fi

# install development dependencies (for testing)
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system -e tests/sonar_test_utils

# The installs above may move packages the KV connector layer relied on.
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ "${PYTORCH_NIGHTLY}" != "1" ] \
        && [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
        python3 tools/kv_connector_lock.py verify requirements/test/kv_connectors_cuda.txt; \
    fi

# enable fast downloads from hf (for testing)
ENV HF_XET_HIGH_PERFORMANCE 1

# increase timeout for hf downloads (for testing)
ENV HF_HUB_DOWNLOAD_TIMEOUT 60

# Copy in the v1 package for testing (it isn't distributed yet)
COPY sonar/v1 /usr/local/lib/python${PYTHON_VERSION}/dist-packages/sonar/v1
#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# base openai image with additional requirements, for any subsequent openai-style images
FROM sonar-base AS sonar-openai-base
ARG TARGETPLATFORM
ARG INSTALL_KV_CONNECTORS=false
ARG EFA_INSTALLER_VERSION
ARG EFA_INSTALLER_SHA256
ARG CUDA_VERSION
ARG TRITON_INSTALL_FROM_SOURCE_REPO

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

# install kv_connectors if requested
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
COPY tools/install_kv_connectors.sh /tmp/install_kv_connectors.sh
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=requirements/kv_connectors.txt,target=/tmp/sonar-requirements/kv_connectors.txt,ro \
    --mount=type=bind,source=requirements/kv_connectors_cu12.txt,target=/tmp/sonar-requirements/kv_connectors_cu12.txt,ro \
    set -eu; \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-'); \
    CUDA_HOME=/usr/local/cuda; \
    # LMCache source builds require CUDA development libraries.
    BUILD_PKGS="libcusparse-dev-${CUDA_VERSION_DASH} \
                libcublas-dev-${CUDA_VERSION_DASH} \
                libcusolver-dev-${CUDA_VERSION_DASH}"; \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        UV_NO_BUILD=true bash /tmp/install_kv_connectors.sh /tmp/sonar-requirements || ( \
            apt-get update -y && \
            apt-get install -y --no-install-recommends --allow-change-held-packages ${BUILD_PKGS} && \
            UV_NO_BUILD_ISOLATION=true bash /tmp/install_kv_connectors.sh /tmp/sonar-requirements && \
            apt-get purge -y ${BUILD_PKGS} && \
            rm -rf /var/lib/apt/lists/* \
        ); \
    fi

# Install the AWS EFA userspace with the KV connectors: NIXL's LIBFABRIC plugin
# and Mooncake's EFA transport load libfabric at runtime.
RUN if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        set -eux; \
        EFA_INSTALLER_ARCHIVE="/tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
        curl -fsSL --retry 3 --retry-all-errors \
            "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
            -o "${EFA_INSTALLER_ARCHIVE}"; \
        echo "${EFA_INSTALLER_SHA256}  ${EFA_INSTALLER_ARCHIVE}" | sha256sum -c -; \
        tar -xzf "${EFA_INSTALLER_ARCHIVE}" -C /tmp; \
        # The installer apt-installs dependencies; earlier layers purged the lists.
        apt-get update -y; \
        # Minimal install: libfabric, aws-ofi-nccl and hwloc only. --build-ngc
        # implies --skip-kmod --skip-limit-conf --no-verify.
        (cd /tmp/aws-efa-installer && \
            ./efa_installer.sh --skip-mpi --skip-rdma-core --build-ngc); \
        test -x /opt/amazon/efa/bin/fi_info; \
        /opt/amazon/efa/bin/fi_info --version; \
        # `-l` lists providers without EFA hardware, so this works in a build.
        /opt/amazon/efa/bin/fi_info -l | grep -q '^efa:'; \
        EFA_NIXL_PLUGIN=$(find /usr/local/lib -name libplugin_LIBFABRIC.so \
            -print -quit); \
        test -n "${EFA_NIXL_PLUGIN}"; \
        ldd "${EFA_NIXL_PLUGIN}" | grep -q 'libfabric\.so\.1 => /opt/amazon/efa/lib/'; \
        # Plugins that NCCL_NET_PLUGIN=ofi and NCCL_TUNER_PLUGIN=ofi resolve to.
        ldconfig -p | grep -q '/opt/amazon/ofi-nccl/lib/libnccl-net-ofi\.so$'; \
        ldconfig -p | grep -q '/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi\.so$'; \
        apt-get clean; \
        rm -rf "${EFA_INSTALLER_ARCHIVE}" /tmp/aws-efa-installer \
            /var/lib/apt/lists/*; \
    fi
# Mooncake registers GPU memory for RDMA via nvidia-peermem by default. Hosts
# without that module loaded (e.g. GB200) must pass WITH_NVIDIA_PEERMEM=0 at run
# time; see docs/features/mooncake_connector_usage.md. Deliberately not an ENV
# here so the image does not override mooncake's own default.

# Install Rubin prerelease dependencies when explicitly requested. Keep this
# after the public Sonar and KV connector dependency pins installed above.
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=requirements/rubin-prerelease.txt,target=/tmp/sonar-requirements/rubin-prerelease.txt,ro \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        echo "Installing Rubin prerelease requirements..." \
        && uv pip install --system \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/sonar-requirements/rubin-prerelease.txt; \
    fi

# Install source-built Triton last so dependency resolution cannot restore the
# previously installed version.
COPY --from=extensions-build \
    /tmp/triton_from_source_workspace/dist/wheel.sha256 \
    /tmp/triton-source-wheel.sha256
COPY tools/install_triton_from_source.sh /tmp/install_triton_from_source.sh
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,from=extensions-build,source=/tmp/triton_from_source_workspace/dist,target=/tmp/triton-source-dist,ro \
    if [ -n "${TRITON_INSTALL_FROM_SOURCE_REPO}" ]; then \
        CUDA_MAJOR_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f1,2); \
        if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
            case "${CUDA_MAJOR_MINOR}" in \
                13.4|13.5) \
                    TRITON_SYSTEM_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
                    TRITON_EXPECTED_PTXAS_VERSION="${CUDA_MAJOR_MINOR}" \
                        bash /tmp/install_triton_from_source.sh \
                            /tmp/triton-source-dist \
                            /tmp/triton-source-wheel.sha256 \
                    ;; \
                *) \
                    echo "Unsupported Rubin CUDA version: ${CUDA_MAJOR_MINOR}" >&2; \
                    exit 1 \
                    ;; \
            esac; \
        else \
            bash /tmp/install_triton_from_source.sh \
                /tmp/triton-source-dist \
                /tmp/triton-source-wheel.sha256; \
        fi; \
    fi \
    && rm -f \
        /tmp/install_triton_from_source.sh \
        /tmp/triton-source-wheel.sha256

ENV SONAR_USAGE_SOURCE production-docker-image

# define sagemaker first, so it is not default from `docker build`
FROM sonar-openai-base AS sonar-sagemaker

COPY examples/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

FROM sonar-openai-base AS sonar-openai

ENTRYPOINT ["sonar", "run"]
#################### OPENAI API SERVER ####################

#################### MAIN WHEEL EXPORT ####################
# Stage to export only the main sonar wheel for distribution.
# Usage:
#   DOCKER_BUILDKIT=1 docker build --target main-wheel-export --output ./dist -f docker/Dockerfile .
FROM scratch AS main-wheel-export
ARG COMMON_WORKDIR=/workspace
COPY --from=build ${COMMON_WORKDIR}/dist/*.whl /
#################### MAIN WHEEL EXPORT ####################
