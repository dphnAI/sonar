---
title: FP8 vision encoder attention
description: Use FP8 attention for Qwen3 vision encoders on supported NVIDIA and AMD GPUs.
---

FP8 attention can reduce vision encoder time for large images and multi-image
requests. Sonar quantizes the query, key, and value tensors before it runs
attention. Small images can be slower because the quantization work can cost
more than it saves.

This feature supports Qwen3 vision transformer models, including Qwen3-VL and
Qwen3.5 variants. Dynamic scaling does not support full CUDA graphs for the
vision encoder.

## Requirements

Use one of these configurations:

- NVIDIA Blackwell: FlashInfer cuDNN attention with cuDNN 9.17.1 or newer.
- AMD MI300 or MI350: AITER with
  `flash_attn_varlen_fp8_pertensor_func`. This corresponds to `gfx942` and
  `gfx950`.

AITER supports packed variable-length image and video batches.

## Enable FP8 attention

Select the backend for your platform.

```sh
# NVIDIA
sonar run "$MODEL" \
  --mm-encoder-attn-backend FLASHINFER \
  --mm-encoder-attn-dtype fp8

# AMD ROCm
sonar run "$MODEL" \
  --mm-encoder-attn-backend ROCM_AITER_FA \
  --mm-encoder-attn-dtype fp8
```

Sonar uses dynamic scaling when you do not give it a scale file. It records the
observed Q/K/V maximum values in a 16-entry circular buffer and updates the
scales during each forward pass.

## Create static scales

Static scales remove the dynamic scaling work from production requests. Create
the scales with representative images.

```sh
sonar bench mm-processor \
  --model "$MODEL" \
  --mm-encoder-attn-backend "$MM_ATTN_BACKEND" \
  --mm-encoder-attn-dtype fp8 \
  --mm-encoder-fp8-scale-save-path /path/to/scales.json \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 100
```

The scale file is written after 16 forward passes. Sonar multiplies the learned
scales by `--mm-encoder-fp8-scale-save-margin`. Its default value is `1.5`.
This margin gives headroom for activation values that were absent from the
calibration data.

Load the saved scales when you start the server.

```sh
sonar run "$MODEL" \
  --mm-encoder-attn-backend "$MM_ATTN_BACKEND" \
  --mm-encoder-attn-dtype fp8 \
  --mm-encoder-fp8-scale-path /path/to/scales.json
```

The scale file has one entry for each vision attention layer:

```json
{
  "visual.blocks.0.attn.attn": {
    "q": 224.0,
    "k": 198.0,
    "v": 210.0
  },
  "visual.blocks.1.attn.attn": {
    "q": 218.0,
    "k": 195.0,
    "v": 207.0
  }
}
```

The keys `q_scale`, `k_scale`, and `v_scale` are accepted as aliases.

## Expected performance

The upstream AITER measurements on an MI300X include FP8 quantization time:

| Sequence length | BF16 | FP8 | Speedup |
| ---: | ---: | ---: | ---: |
| 2,304 | 0.467 ms | 0.337 ms | 1.38x |
| 4,096 | 0.812 ms | 0.764 ms | 1.06x |
| 8,192 | 2.555 ms | 2.364 ms | 1.08x |
| 16,384 | 9.769 ms | 8.655 ms | 1.13x |

NVIDIA measurements show that the crossover depends on the image count and
resolution. Test your model and request distribution before you enable this
feature in production.
