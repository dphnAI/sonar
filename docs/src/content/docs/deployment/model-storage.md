---
title: Model loading and storage
description: Control model downloads, revisions, caches, and weight loading.
---

Sonar accepts a Hugging Face repository name or a local model directory. Local
directories must contain the model configuration, tokenizer files, and weight
files required by the selected architecture.

## Authenticate to Hugging Face

```bash
export HF_TOKEN="hf_..."
sonar serve ORGANIZATION/PRIVATE_MODEL
```

Give the token read access only. Do not put it in a command argument or image
layer.

## Pin a revision

```bash
sonar serve ORGANIZATION/MODEL \
  --revision MODEL_COMMIT \
  --tokenizer-revision TOKENIZER_COMMIT
```

A branch or tag can change. Use commit identifiers when a deployment must be
reproducible.

## Select cache storage

Use `HF_HOME` to place the Hugging Face cache on persistent storage:

```bash
export HF_HOME=/srv/sonar/huggingface
sonar serve ORGANIZATION/MODEL
```

The cache needs enough temporary space for concurrent downloads and extracted
metadata. Keep it outside a small container writable layer.

Use `--download-dir` when Sonar should use a specific model download location.

## Prepare an offline host

Download and verify the complete model on a connected machine. Copy the
snapshot to the offline host and pass its local path:

```bash
sonar serve /models/qwen3-8b
```

Include tokenizer files and custom code. Test startup with network access
disabled before you move the image into production.

## Choose a load format

The default `auto` format selects a compatible loader from the checkpoint.
Safetensors is the normal Hugging Face format. Other loaders can improve
startup on a supported storage system.

Set a load format only when the checkpoint or storage stack requires it:

```bash
sonar serve MODEL --load-format safetensors
```

Use the [server argument reference](/reference/server-arguments/) to inspect
available loaders and their dependencies.

## Diagnose slow loading

Separate download time from weight loading time. A populated cache removes the
network transfer but does not remove disk reads or GPU copies.

Check local read bandwidth, network filesystem contention, safetensors shard
count, and free temporary space. Preserve the compilation cache because native
kernel compilation occurs after weights load.
