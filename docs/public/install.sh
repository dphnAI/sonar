#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

set -eu

SONAR_BASE_URL="${SONAR_BASE_URL:-https://sonar.dphn.ai}"
SONAR_UV_VERSION="${SONAR_UV_VERSION:-0.11.32}"

assume_yes=0
backend=auto
channel=
venv_path=
python_version=3.13
dry_run=0
skip_platform_checks=0

say() {
    printf '%s\n' "$*"
}

warn() {
    printf 'Warning: %s\n' "$*" >&2
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Install Sonar into a virtual environment.

Usage:
  install.sh [options]

Options:
  --yes                       Accept prompts and use defaults.
  --channel release|nightly   Select the release channel.
  --backend auto|cuda|rocm|xpu|metal|cpu
                              Override hardware detection.
  --venv PATH                 Use or create this virtual environment.
  --python VERSION            Python version for a new environment (default: 3.13).
  --dry-run                   Print commands without changing the system.
  --skip-platform-checks      Skip driver and hardware compatibility checks.
  -h, --help                  Show this help.

Environment overrides:
  SONAR_BASE_URL              Site that hosts Sonar platform wheels.
  SONAR_UV_VERSION            uv installer version.
EOF
}

shell_quote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

run() {
    if [ "$dry_run" -eq 1 ]; then
        printf '+'
        for argument in "$@"; do
            printf ' %s' "$(shell_quote "$argument")"
        done
        printf '\n'
        return 0
    fi
    "$@"
}

has_tty() {
    [ -r /dev/tty ] && [ -w /dev/tty ]
}

ask_yes_no() {
    prompt=$1
    default=${2:-yes}

    if [ "$assume_yes" -eq 1 ]; then
        [ "$default" = yes ]
        return
    fi
    has_tty || die "This choice requires a terminal. Rerun with --yes or explicit options."

    if [ "$default" = yes ]; then
        suffix='[Y/n]'
    else
        suffix='[y/N]'
    fi

    while :; do
        printf '%s %s ' "$prompt" "$suffix" >/dev/tty
        IFS= read -r answer </dev/tty || die "Could not read your answer."
        case "$answer" in
            y | Y | yes | YES | Yes) return 0 ;;
            n | N | no | NO | No) return 1 ;;
            "")
                [ "$default" = yes ]
                return
                ;;
        esac
        say "Enter yes or no."
    done
}

ask_value() {
    prompt=$1
    default=$2
    has_tty || die "This choice requires a terminal. Pass the corresponding command-line option."
    printf '%s [%s]: ' "$prompt" "$default" >/dev/tty
    IFS= read -r answer </dev/tty || die "Could not read your answer."
    if [ -n "$answer" ]; then
        printf '%s\n' "$answer"
    else
        printf '%s\n' "$default"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --yes)
            assume_yes=1
            shift
            ;;
        --channel)
            [ "$#" -ge 2 ] || die "--channel requires a value."
            channel=$2
            shift 2
            ;;
        --backend)
            [ "$#" -ge 2 ] || die "--backend requires a value."
            backend=$2
            shift 2
            ;;
        --venv)
            [ "$#" -ge 2 ] || die "--venv requires a path."
            venv_path=$2
            shift 2
            ;;
        --python)
            [ "$#" -ge 2 ] || die "--python requires a version."
            python_version=$2
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --skip-platform-checks)
            skip_platform_checks=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

case "$channel" in
    "" | release | nightly) ;;
    *) die "Unsupported channel '$channel'. Use release or nightly." ;;
esac

case "$backend" in
    auto | cuda | rocm | xpu | metal | cpu) ;;
    *) die "Unsupported backend '$backend'." ;;
esac

os=$(uname -s)
machine=$(uname -m)
case "$machine" in
    x86_64 | amd64) architecture=x86_64 ;;
    aarch64 | arm64) architecture=aarch64 ;;
    *) die "Sonar does not publish wheels for architecture '$machine' yet." ;;
esac

detect_backend() {
    if [ "$os" = Darwin ] && [ "$architecture" = aarch64 ]; then
        printf '%s\n' metal
    elif [ "$os" = Linux ] && command -v nvidia-smi >/dev/null 2>&1 &&
        nvidia-smi -L >/dev/null 2>&1; then
        printf '%s\n' cuda
    elif [ "$os" = Linux ] && {
        command -v rocminfo >/dev/null 2>&1 || [ -e /dev/kfd ]
    }; then
        printf '%s\n' rocm
    elif [ "$os" = Linux ] && {
        command -v xpu-smi >/dev/null 2>&1 || command -v sycl-ls >/dev/null 2>&1
    }; then
        printf '%s\n' xpu
    else
        printf '%s\n' cpu
    fi
}

if [ "$backend" = auto ]; then
    backend=$(detect_backend)
fi

case "$backend:$os:$architecture" in
    cuda:Linux:x86_64 | rocm:Linux:x86_64 | xpu:Linux:x86_64) ;;
    cpu:Linux:x86_64 | cpu:Linux:aarch64 | metal:Darwin:aarch64) ;;
    *)
        die "Sonar does not publish a $backend wheel for $os/$architecture."
        ;;
esac

version_at_least() {
    current=$1
    required=$2
    first=$(printf '%s\n%s\n' "$required" "$current" | sort -V | head -n 1)
    [ "$first" = "$required" ]
}

check_cuda() {
    command -v nvidia-smi >/dev/null 2>&1 ||
        die "nvidia-smi is required to validate the NVIDIA driver."
    nvidia-smi -L >/dev/null 2>&1 || die "The NVIDIA driver cannot access a GPU."

    supported_cuda=$(nvidia-smi 2>/dev/null |
        sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' |
        head -n 1)
    [ -n "$supported_cuda" ] ||
        die "Could not determine the CUDA version supported by the NVIDIA driver."
    version_at_least "$supported_cuda" 13.0 ||
        die "The NVIDIA driver supports CUDA $supported_cuda. Sonar requires CUDA 13.0 or newer."

    compute_caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
    [ -n "$compute_caps" ] ||
        die "Could not determine the NVIDIA GPU compute capability."
    unsupported_caps=
    old_ifs=$IFS
    IFS='
'
    for capability in $compute_caps; do
        case "$capability" in
            8.0 | 8.6 | 8.9 | 9.0 | 10.0 | 10.3 | 12.*) ;;
            *) unsupported_caps="${unsupported_caps}${unsupported_caps:+, }${capability}" ;;
        esac
    done
    IFS=$old_ifs
    [ -z "$unsupported_caps" ] ||
        die "The CUDA wheel does not contain kernels for compute capability: $unsupported_caps."
}

if [ "$skip_platform_checks" -eq 0 ]; then
    case "$backend" in
        cuda) check_cuda ;;
        rocm)
            [ -e /dev/kfd ] || die "No AMD KFD device was found at /dev/kfd."
            ;;
        xpu)
            command -v xpu-smi >/dev/null 2>&1 || command -v sycl-ls >/dev/null 2>&1 ||
                die "Install the Intel GPU runtime before installing the XPU wheel."
            ;;
        metal)
            [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 14 ] ||
                die "The Metal wheel requires macOS 14 or newer."
            ;;
    esac
fi

library_available() {
    library=$1
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "$library"; then
        return 0
    fi
    for directory in /lib /lib64 /usr/lib /usr/lib64 /usr/local/lib; do
        [ -e "$directory/$library" ] && return 0
    done
    return 1
}

detect_package_manager() {
    for manager in apt-get dnf yum zypper pacman; do
        if command -v "$manager" >/dev/null 2>&1; then
            printf '%s\n' "$manager"
            return
        fi
    done
    printf '%s\n' unknown
}

install_cpu_system_dependencies() {
    library_available libnuma.so.1 && return

    manager=$(detect_package_manager)
    case "$manager" in
        apt-get) install_command="apt-get update && apt-get install -y libnuma1" ;;
        dnf) install_command="dnf install -y numactl-libs" ;;
        yum) install_command="yum install -y numactl-libs" ;;
        zypper) install_command="zypper --non-interactive install libnuma1" ;;
        pacman) install_command="pacman -S --needed --noconfirm numactl" ;;
        *)
            die "libnuma.so.1 is missing. Install your distribution's libnuma runtime package and rerun this installer."
            ;;
    esac

    say "The CPU wheel requires libnuma.so.1."
    if [ "$(id -u)" -eq 0 ]; then
        # shellcheck disable=SC2086
        run sh -c "$install_command"
    elif command -v sudo >/dev/null 2>&1; then
        if ask_yes_no "Install the required package with sudo?" yes; then
            # shellcheck disable=SC2086
            run sudo sh -c "$install_command"
        else
            say "Run this command, then rerun the installer:"
            say "  sudo $install_command"
            exit 1
        fi
    else
        say "sudo is not available. Run this command as root, then rerun the installer:"
        say "  $install_command"
        exit 1
    fi

    [ "$dry_run" -eq 1 ] || library_available libnuma.so.1 ||
        die "libnuma.so.1 is still unavailable after package installation."
}

if [ "$backend" = cpu ]; then
    install_cpu_system_dependencies
fi

if [ "$os" = Linux ] && [ "$skip_platform_checks" -eq 0 ]; then
    glibc_version=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)
    [ -n "$glibc_version" ] ||
        die "Could not determine the system glibc version."
    version_at_least "$glibc_version" 2.35 ||
        die "The published Linux wheels require glibc 2.35 or newer; this system has $glibc_version."
fi

say "Detected platform: $os $architecture, backend: $backend"

if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || die "curl is required to install uv."
    if ask_yes_no "uv is not installed. Install uv $SONAR_UV_VERSION?" yes; then
        if [ "$dry_run" -eq 1 ]; then
            say "+ curl -LsSf https://astral.sh/uv/$SONAR_UV_VERSION/install.sh | sh"
        else
            uv_installer="${TMPDIR:-/tmp}/sonar-uv-install-$$.sh"
            trap 'rm -f "$uv_installer"' EXIT HUP INT TERM
            curl -LsSf "https://astral.sh/uv/$SONAR_UV_VERSION/install.sh" \
                -o "$uv_installer"
            sh "$uv_installer"
            rm -f "$uv_installer"
            trap - EXIT HUP INT TERM
            PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
            export PATH
        fi
    else
        die "Install uv from https://docs.astral.sh/uv/ and rerun this installer."
    fi
fi

if [ "$dry_run" -eq 0 ]; then
    command -v uv >/dev/null 2>&1 || die "uv was installed but is not available on PATH."
fi

if [ -z "$venv_path" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        if ask_yes_no "Use the active virtual environment at $VIRTUAL_ENV?" yes; then
            venv_path=$VIRTUAL_ENV
        fi
    fi
fi

if [ -z "$venv_path" ]; then
    if [ "$assume_yes" -eq 1 ] || ask_yes_no "Create a virtual environment at $HOME/.sonar_venv?" yes; then
        venv_path=$HOME/.sonar_venv
    else
        venv_path=$(ask_value "Path to an existing virtual environment" "$HOME/.venv")
    fi
fi

case "$venv_path" in
    \~) venv_path=$HOME ;;
    \~/*) venv_path=$HOME/${venv_path#\~/} ;;
esac

if [ ! -x "$venv_path/bin/python" ]; then
    [ ! -e "$venv_path" ] || [ -d "$venv_path" ] ||
        die "$venv_path exists and is not a directory."
    if [ -d "$venv_path" ] &&
        [ -n "$(find "$venv_path" ! -path "$venv_path" -print -quit 2>/dev/null)" ]; then
        die "$venv_path is not a compatible virtual environment and is not empty."
    fi
    run uv venv "$venv_path" --python "$python_version" --seed --prompt sonar
fi

venv_python=$venv_path/bin/python
if [ "$dry_run" -eq 0 ]; then
    [ -f "$venv_path/pyvenv.cfg" ] || die "$venv_path is not a Python virtual environment."
    "$venv_python" - "$architecture" <<'PY'
import platform
import sys

expected_arch = sys.argv[1]
if sys.implementation.name != "cpython":
    raise SystemExit("Sonar wheels require CPython.")
if not ((3, 11) <= sys.version_info[:2] < (3, 15)):
    raise SystemExit(
        f"Sonar requires Python 3.11-3.14; this environment uses "
        f"{platform.python_version()}."
    )
machine = platform.machine().lower()
normalized = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}.get(machine, machine)
if normalized != expected_arch:
    raise SystemExit(
        f"The environment uses architecture {machine}, but the selected wheel "
        f"uses {expected_arch}."
    )
PY
fi

if [ -z "$channel" ]; then
    if [ "$assume_yes" -eq 1 ]; then
        channel=release
    else
        channel=$(ask_value "Install channel (release or nightly)" release)
    fi
fi
case "$channel" in
    release | nightly) ;;
    *) die "Unsupported channel '$channel'. Use release or nightly." ;;
esac

set -- uv pip install --python "$venv_python" --upgrade aphrodite-engine
case "$backend:$channel" in
    cuda:release)
        set -- "$@" --torch-backend cu130
        ;;
    cuda:nightly)
        set -- "$@" \
            --extra-index-url "$SONAR_BASE_URL/nightly" \
            --index-strategy first-index \
            --torch-backend cu130
        ;;
    *)
        wheel_index="$SONAR_BASE_URL/whl/$channel/$backend/$architecture/simple"
        if [ "$dry_run" -eq 0 ]; then
            curl --fail --location --silent --show-error \
                "${wheel_index}/aphrodite-engine/" \
                -o /dev/null ||
                die "No $channel wheel index is available for $backend/$architecture yet."
        fi
        set -- "$@" \
            --extra-index-url "$wheel_index" \
            --index-strategy first-index
        if [ "$backend" = cpu ]; then
            # triton-cpu is a direct URL dependency, which wheels cannot declare for uv, so
            # request it here. Keep the URLs and markers in sync with requirements/cpu.txt.
            triton_cpu=https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0+cpu
            set -- "$@" --torch-backend cpu \
                "triton @ $triton_cpu-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ; platform_machine == 'x86_64' and python_version == '3.11'" \
                "triton @ $triton_cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ; platform_machine == 'x86_64' and python_version >= '3.12'"
        fi
        ;;
esac

say "Installing Sonar from the $channel channel."
run "$@"

if [ "$dry_run" -eq 0 ]; then
    "$venv_python" - "$backend" "$channel" <<'PY'
import json
import pathlib
import sys

import aphrodite

record = {
    "backend": sys.argv[1],
    "channel": sys.argv[2],
    "version": getattr(aphrodite, "__version__", "unknown"),
}
path = pathlib.Path(sys.prefix) / ".sonar-install.json"
path.write_text(json.dumps(record, indent=2) + "\n")
print(f"Installed Sonar {record['version']}.")
PY
fi

say
say "Activate the environment with:"
say "  . $(shell_quote "$venv_path/bin/activate")"
say
say "Then verify the installation with:"
say "  aphrodite --version"
