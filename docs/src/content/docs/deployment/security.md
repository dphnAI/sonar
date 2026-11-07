---
title: Security
description: Reduce risk when Sonar serves untrusted clients or models.
---

Treat model files, remote code, media URLs, prompt content, and adapters as
untrusted inputs unless you control their source.

## Limit network exposure

Use `--api-key` for bearer authentication. Place public deployments behind a
proxy that provides TLS, rate limits, and request-size limits.

Do not expose diagnostic or administrative routes to the public network.
Restrict `/metrics`, profiling routes, adapter management, cache reset, sleep,
weight update, and collective RPC routes.

## Pin model revisions

Use `--revision` and `--tokenizer-revision` with immutable commit identifiers.
Review a model repository before you set `--trust-remote-code`.

Remote model code executes in the server process. It receives the same file,
network, and device access as Sonar.

## Restrict multimodal media

Allow only required remote hosts:

```bash
sonar serve MODEL \
  --allowed-media-domains images.example.com assets.example.net
```

Leave `--allowed-local-media-path` empty for a public service. If local files
are required, point it at a dedicated read-only directory. Never allow `/`.

The proxy should limit media size and request duration. Remote media fetching
can otherwise access slow or unexpectedly large resources.

## Protect adapter management

Dynamic LoRA loading changes server state and reads a model path. Put the load
and unload routes on an administrative network. Use an allowlist for adapter
locations.

## Isolate tenants

Use separate replicas when tenants require strong cache or resource isolation.
Shared prefix caches, queues, logs, and metrics can reveal workload
characteristics even when response data remains separate.

## Container controls

Run without privileged mode. Mount model storage read-only when possible. Give
the process a writable cache directory and no access to unrelated host paths.

The container still needs GPU device access and shared memory. These
requirements do not require broad host filesystem access.

## Secure Ray clusters

Sonar treats an entire Ray cluster as one trust domain. A principal that can
submit actors or tasks can execute arbitrary code on any cluster node and is
therefore as trusted as the driver and API server. Environment variables do not
form a security boundary inside the cluster. This matches
[Ray's security model](https://docs.ray.io/en/latest/ray-core/security.html).

`RayExecutorV2` copies the driver's environment to remote workers so they
receive required Sonar settings, communication-library tuning, and credentials
used to download gated models. `get_driver_env_vars()` in
`sonar/v1/executor/ray_env_utils.py` uses a copy-all-except-denylist
policy: it copies every variable except worker-specific variables and variables
that the operator excludes. Workers apply copied values with `setdefault`, so
an existing worker-side value is not overwritten.

This behavior also copies credentials such as Hugging Face tokens, cloud
storage keys, registry tokens, and internal service tokens when they are
present in the driver's environment. A process running as the same OS user on
a worker may be able to read them through `/proc/<pid>/environ`.

To exclude variables, create
`$SONAR_CONFIG_ROOT/ray_non_carry_over_env_vars.json` (by default,
`~/.config/sonar/ray_non_carry_over_env_vars.json`) containing their names:

```json
[
  "HF_TOKEN",
  "AWS_SECRET_ACCESS_KEY",
  "AWS_SESSION_TOKEN",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "AZURE_CLIENT_SECRET",
  "REGISTRY_TOKEN",
  "MY_INTERNAL_SERVICE_KEY"
]
```

Also minimize the driver's environment by loading credentials from a secrets
manager or mounted file when possible. If workers must have less trust than the
driver, use separate OS users or containers with non-overlapping UIDs and
isolate `/proc` between them. On Linux hosts, mounting `procfs` with `hidepid=2`
can prevent same-UID processes from reading another process's environment.

Restrict Ray cluster membership to trusted principals. Use Ray TLS
authentication, isolate the cluster network, and do not expose the Ray client
port or dashboard to untrusted networks.

## Log handling

Prompts and generated text can contain secrets. Check log settings before
production use. Restrict access to traces and request logs, and apply a
retention period.
