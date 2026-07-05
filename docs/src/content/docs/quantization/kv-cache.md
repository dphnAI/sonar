---
title: KV Cache Quantization
---

Aphrodite supports FP8 KV cache quantization method. We support two modes: FP8_E5M2 and FP8_E4M3.

## FP8_E5M2

This FP8 format retains 5 bits of exponent and 2 bits of mantissa. It is a good balance between precision and performance.

You can simply load a model with the `--kv-cache-dtype fp8` flag to enable this.

## FP8_E4M3

Quantizing the KV cache to FP8 reduces its memory footprint. This increases the number of tokens that can be stored in the cache, improving throughput. OCP (Open Compute Project <www.opencompute.org>) specifies two common 8-bit floating point data formats: E5M2 (5 exponent bits and 2 mantissa bits) and E4M3FN (4 exponent bits and 3 mantissa bits), often shortened as E4M3. One benefit of the E4M3 format over E5M2 is that floating point numbers are represented in higher precision. However, the small dynamic range of FP8 E4M3 (±240.0 can be represented) typically necessitates the use of a higher-precision (typically FP32) scaling factor alongside each quantized tensor. For now, only per-tensor (scalar) scaling factors are supported. Development is ongoing to support scaling factors of a finer granularity (e.g. per-channel).

These scaling factors can be specified by passing an optional quantization param JSON to the LLM engine at load time. If this JSON is not specified, scaling factors default to 1.0. These scaling factors are typically obtained when running an unquantized model through a quantizer tool (e.g. AMD quantizer or NVIDIA AMMO).

To install AMMO (AlgorithMic Model Optimization):

```sh
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo
```

Studies have shown that FP8 E4M3 quantization typically only minimally degrades inference accuracy. The most recent silicon offerings e.g. AMD MI300, NVIDIA Hopper or later support native hardware conversion to and from fp32, fp16, bf16, etc. Thus, LLM inference is greatly accelerated with minimal accuracy loss.

Please refer to [this example](https://github.com/PygmalionAI/aphrodite-engine/blob/main/examples/fp8/README.md) to generate kv_cache_scales.json of your own.

Here is an example of how to load a model with FP8 KV cache quantization:

```sh
from aphrodite import LLM, SamplingParams
sampling_params = SamplingParams(temperature=1.3, top_p=0.8)
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
          kv_cache_dtype="fp8",
          quantization_param_path="./llama2-7b-fp8-kv/kv_cache_scales.json")
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

Or you can use the CLI:

```sh
aphrodite run meta-llama/Llama-2-7b-chat-hf --kv-cache-dtype fp8 --quantization-param-path ./llama2-7b-fp8-kv/kv_cache_scales.json
```

## Skipping Specific Layers from KV Cache Quantization

Some attention layer types, such as sliding-window attention, are more
sensitive to KV cache quantization. The `--kv-cache-dtype-skip-layers` flag
leaves the specified layers at the model's native dtype while keeping the rest
of the layers under the chosen quantized dtype. The flag accepts layer indices
or layer-type names:

```sh
aphrodite run <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window

aphrodite run <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers 0 1 23
```

Programmatic usage:

```python
from aphrodite import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["sliding_window"],
)
```

Note, current prefix caching doesn't work with FP8 KV cache enabled, forward_prefix kernel should handle different KV and cache type.
