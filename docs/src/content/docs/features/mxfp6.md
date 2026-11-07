---
title: Online MXFP6
description: Convert BF16 checkpoints to native MXFP6 while Sonar loads them on NVIDIA Thor.
---

Sonar can convert BF16 or FP16 weights to OCP MXFP6 while it loads a model. The initial native backend targets NVIDIA Thor GPUs with compute capability 11.0. It does not write a converted checkpoint to disk.

## Start a model

Use `--quantization mxfp6` to select E2M3 weights and dynamic MXFP8 activations:

```sh
sonar run Qwen/Qwen3.5-4B --quantization mxfp6
```

The loader converts one layer at a time. It replaces each full-precision tensor with packed six-bit data and an E8M0 scale for each block of 32 values. This limits temporary memory use. The model does not have to fit in host memory twice.

Sonar keeps router gates and shared-expert gates in BF16 by default. These small tensors are sensitive to quantization and do not account for much model memory.

## Select the FP6 encoding

The shorthand uses E2M3 weights. Use `--quantization online` with a JSON configuration to select E3M2:

```sh
sonar run Qwen/Qwen3.5-4B \
  --quantization online \
  --quantization-config '{
    "linear":{"weight":"mxfp6_e3m2","activation":"mxfp8"},
    "moe":{"weight":"mxfp6_e3m2","activation":"mxfp8"}
  }'
```

E2M3 has more mantissa precision. E3M2 has a larger finite range. Start with E2M3 unless evaluation shows that the model needs the E3M2 range.

## Use W6A6

The native backend also accepts MXFP6 activations. This mode reduces activation precision and adds conversion work:

```sh
sonar run Qwen/Qwen3.5-4B \
  --quantization online \
  --quantization-config '{
    "linear":{"weight":"mxfp6_e2m3","activation":"mxfp6_e2m3_dynamic"},
    "moe":{"weight":"mxfp6_e2m3","activation":"mxfp6_e2m3_dynamic"}
  }'
```

Benchmark W6A6 against the default W6A8 mode on the target workload. W6A8 is faster on the current Thor kernel.

## Keep selected modules in BF16

Use ordered overrides to control precision. Sonar applies the last matching rule:

```sh
sonar run Qwen/Qwen3.5-4B \
  --quantization mxfp6 \
  --quantization-config '{
    "overrides":[
      {"pattern":"re:.*\\.lm_head$","weight":"bf16"},
      {"pattern":"model.layers.0.mlp.down_proj","weight":"bf16"}
    ]
  }'
```

An exact module prefix matches that module. A pattern that starts with `re:` is a regular expression. Later rules can refine an earlier broad rule.

## Current limits

- Native execution requires CUDA, compute capability 11.0, and `nvidia-cutlass-dsl`.
- Dense input and output dimensions must be multiples of 128.
- MoE hidden and intermediate dimensions must be multiples of 128.
- The native MoE path supports gated SiLU without expert bias or SwiGLU clamping.
- Expert parallel execution stays in BF16. Tensor parallel execution is supported.
- Unsupported layers stay in BF16 and produce a startup warning.

Evaluate model quality before production use. Online conversion has no calibration dataset and can affect different models in different ways.
