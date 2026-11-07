---
title: Upgrade Sonar
description: Upgrade Sonar while preserving a working rollback.
---

Test an upgrade with the production model and configuration before you replace
a running replica.

## Migrate from Aphrodite

Install `sonar` in a fresh virtual environment using the
[installer or release index](/getting-started/installation/). Keep the previous
environment available for rollback. The old `aphrodite-engine` PyPI package
does not install the renamed release.

Do not install `sonar` alongside `aphrodite-engine`: both own the `aphrodite`
command and import package, so uninstalling either can remove the other's files.

Update application imports from `aphrodite` to `sonar`, CLI invocations from
`aphrodite` to `sonar`, and environment variables from `APHRODITE_*` to
`SONAR_*`. Python identifiers also change, for example `AphroditeConfig` becomes
`SonarConfig`. Plugin authors must change entry-point groups from `aphrodite.*`
to `sonar.*` and rebuild native extensions against the renamed package.

Third-party connectors and plugins need versions compatible with the `sonar`
namespace. An external integration named `aphrodite` is not automatically
compatible; check its provider's support before migrating that deployment.

LM Format Enforcer uses its framework-neutral tokenizer API and does not
require an external Aphrodite adapter.

Rebuild container images from this source before deploying the renamed CLI.
The EFA Dockerfile expects a Sonar base image built with
`INSTALL_KV_CONNECTORS=true`, not a pre-migration Aphrodite image.

Update deployment configuration and clients that use engine-specific names,
including `sonar_xargs`, `--model-impl sonar`, `--generation-config sonar`,
and the `sonar:` Prometheus metric prefix. Rebuild dashboards and recording
rules that refer to the old metric names.

Regenerate clients for the Rust frontend's optional `--grpc-port` listener
from the renamed `sonar` protobuf package. The separate Python `--grpc`
server retains its external `smg_grpc_proto` protocol.

The temporary `aphrodite` CLI forwards to Sonar and warns at startup to switch
to `sonar`. Importing `aphrodite` raises an error directing you to the renamed
Python API; it does not alias that API. There are no `APHRODITE_*`
environment-variable fallbacks. Existing Aphrodite native wheels and serialized
Python objects are not interchangeable with the renamed package.

## Record the current deployment

Save the Sonar version, commit, model revision, tokenizer revision, PyTorch
version, CUDA or ROCm version, and complete server arguments.

Export a small set of representative requests with expected output properties.
Exact generated text can vary because of floating-point and batching changes.

## Upgrade a wheel installation

```bash
uv pip install --upgrade sonar \
  --extra-index-url https://sonar.dphn.ai/whl/release/cuda/x86_64/simple \
  --index-strategy first-index \
  --torch-backend cu130
```

For a nightly build:

```bash
uv pip install --upgrade sonar \
  --extra-index-url https://sonar.dphn.ai/whl/nightly/cuda/x86_64/simple \
  --index-strategy first-index \
  --torch-backend cu130
```

On Linux ARM64, replace `cuda/x86_64` with `cuda/aarch64` in either index URL.

Use a fresh virtual environment when core dependencies change. Keep the old
environment until the replacement passes validation.

## Upgrade a source checkout

```bash
git fetch origin
git switch main
git pull --ff-only
uv pip install --editable . --torch-backend=cu130
```

Regenerate the CMake preset when Python, CUDA, the compiler, or the virtual
environment changes. Rebuild native extensions after CMake or `csrc/` changes.

## Validate compatibility

Check startup warnings for renamed flags and deprecated values. Compare the
generated [server argument reference](/reference/server-arguments/) with the
saved command.

Run chat, streaming, structured output, tool calling, LoRA, and multimodal
requests that the service supports. Compare latency and memory with the saved
baseline.

## Roll back

Keep the old image or virtual environment and its model revision. Remove the
failed replacement from traffic, then restore the old replica.

Do not reuse a compilation cache across incompatible PyTorch or CUDA builds
when a rollback produces loader or symbol errors. Use separate cache keys for
each build stack.
