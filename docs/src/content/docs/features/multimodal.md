---
title: Multimodal inputs
description: Send images, audio, and video to supported models.
---

Select an architecture from the
[multimodal model registry](/reference/models/#multimodal). Each model defines
its accepted modalities, chat template, and preprocessing limits.

## Send an image

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://assets.example.com/image.jpg"},
                },
            ],
        }
    ],
)
```

Use a data URL only for small inputs. Base64 increases request size and proxy
memory use.

## Restrict remote media

```bash
sonar serve MODEL \
  --allowed-media-domains assets.example.com images.example.net
```

Allow only hosts that you control. Remote fetching increases request latency
and can access unexpectedly large files.

Use `--allowed-local-media-path` only for a trusted application. Point it at a
dedicated read-only directory.

## Limit items per prompt

```bash
sonar serve MODEL \
  --limit-mm-per-prompt '{"image": 4, "video": 1, "audio": 1}'
```

Limits protect encoder memory and preprocessing capacity. The model processor
can impose a lower limit.

## Configure preprocessing

`--mm-processor-kwargs` passes model-specific settings to the processor. Use
settings documented by the model implementation. Image resolution, video frame
count, and audio duration affect both latency and memory.

## Configure video decoding

Sonar decodes video bytes into frames using a selectable decoding backend.

| Backend | Device | Description |
| --- | --- | --- |
| `opencv` (default) | CPU | OpenCV-based decoder |
| `pyav` | CPU | PyAV decoder |
| `torchcodec` | CPU | TorchCodec (PyTorch-native) decoder |
| `pynvvideocodec` | GPU | NVIDIA PyNvVideoCodec decoder |
| `deepstream` | GPU | NVIDIA DeepStream decoder |

The three CPU backends are ultimately backed by FFmpeg. `torchcodec` lets you
choose which FFmpeg version is used, while `opencv` and `pyav` rely on the
FFmpeg build they were linked against.

Select the decoder with `--media-io-kwargs`:

```bash
sonar serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "torchcodec"}}'
```

### GPU video decoding with PyNvVideoCodec

The `pynvvideocodec` backend uses NVIDIA NVDEC to decode sampled video frames
on the GPU before copying them into host memory for multimodal preprocessing.
For workloads with large videos and relatively light inference, such as video
tagging, this can alleviate bottlenecks in CPU-based video decoders.

> **Warning:** [CUDA Multi-Process Service
> (MPS)](https://docs.nvidia.com/deploy/mps/quick-start.html) is required when
> using this backend. Video decoding runs in the API server process while model
> serving runs in the engine process, so multiple CUDA processes share the same
> GPU. Configure and start MPS before starting Sonar.

You must also set a positive `--mm-ipc-gpu-memory-gb` value to reserve VRAM for
video decoding. Sonar carves this budget out of the memory available to the
KV cache and uses it to bound concurrent frontend decode allocations. If the
budget is exhausted, decode work waits instead of consuming the engine's VRAM
headroom and potentially causing an out-of-memory error while serving requests.

Select the backend with an environment variable and specify a
workload-appropriate VRAM budget. For example, to reserve 1 GiB:

```bash
export SONAR_VIDEO_LOADER_BACKEND=pynvvideocodec
sonar serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --mm-ipc-gpu-memory-gb 1
```

Alternatively, select it with `--media-io-kwargs`:

```bash
sonar serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "pynvvideocodec"}}' \
  --mm-ipc-gpu-memory-gb 1
```

Choose a budget large enough for the largest sampled video that a single API
server process must decode. When using multiple API server processes,
Sonar divides the configured budget evenly among them.

For streaming video sources, use the `deepstream` backend instead.

## Benchmark media

Text-only tests do not measure encoder or download cost. Use
`sonar bench mm-processor` for preprocessing and a multimodal serving
dataset for end-to-end latency.

Warm media caches before a warm-cache test. Keep a separate cold-media result.

## Troubleshoot inputs

Check the model chat template when text works and media fails. Confirm MIME
type, URL access from the server, item count, and processor limits.

Set a request body limit at the proxy. Return a client error before the server
downloads media that violates policy.
