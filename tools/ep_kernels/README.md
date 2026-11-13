# Expert parallel kernels

Large-scale cluster-level expert parallel, as described in the [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437), is an efficient way to deploy sparse MoE models with many experts. However, such deployment requires many components beyond a normal Python package, including system package support and system driver support. It is impossible to bundle all these components into a Python package.

Here we break down the requirements in 2 steps:

1. Build and install the [DeepEP](https://github.com/deepseek-ai/DeepEP) Python library, including necessary dependencies like NVSHMEM. This step does not require any privileged access. Any user can do this.
2. Configure NVIDIA driver to enable IBGDA. This step requires root access, and must be done on the host machine.

2 is necessary for multi-node deployment.

The Python library installer accepts `--workspace <dir>` for staging the build, defaulting to `$(pwd)/ep_kernels_workspace`.

## NCCL version requirement for DeepEPv2

DeepEPv2 (`--all2all-backend deepep_v2`) uses the NCCL GIN (GPU-Initiated
Networking) backend, which requires NCCL >= 2.30.4 both when DeepEP is built
and at runtime. If PyTorch pins an older NCCL release, a plain `pip install`
or `uv pip install` of Sonar is not enough.

Upgrade NCCL before running `install_python_libraries.sh`. Use
`nvidia-nccl-cu12` on CUDA 12.

**With uv** (recommended):

```bash
echo "nvidia-nccl-cu13>=2.30.4" > /tmp/nccl-override.txt
export UV_OVERRIDE=/tmp/nccl-override.txt
uv pip install sonar \
  --extra-index-url https://sonar.dphn.ai/whl/release/cuda/x86_64/simple \
  --index-strategy first-index \
  --torch-backend cu132
```

Keep `UV_OVERRIDE` set for later installs into the same environment. Without
it, any `uv pip install` that resolves PyTorch again restores the pinned NCCL.

**With pip**, or to fix an existing environment:

Install Sonar using the [installer or wheel index](https://sonar.dphn.ai/getting-started/installation/)
first, then override its NCCL dependency:

```bash
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

The override must happen before building DeepEP and remain in place at runtime.
`install_python_libraries.sh` warns when the installed NCCL is too old. If
DeepEP was built against an older NCCL, upgrade NCCL and run the script again.
When NCCL is too old at runtime, Sonar logs the reason at startup and
`deepep_v2` fails with the same message. You can check with:

```bash
python -c "from sonar.utils.import_utils import deep_ep_v2_unavailable_reason; print(deep_ep_v2_unavailable_reason() or 'deepep_v2 is available')"
```

## Usage

```bash
# for hopper
TORCH_CUDA_ARCH_LIST="9.0" bash install_python_libraries.sh
# for blackwell
TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh
```

Additional step for multi-node deployment:

```bash
sudo bash configure_system_drivers.sh
sudo reboot # Reboot is required to load the new driver
```
