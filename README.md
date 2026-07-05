<h1 align="center">
Breathing Life into Language
</h1>

![sonar](assets/sonar.jpg)

Sonar (formerly Aphrodite Engine) is an inference engine that optimizes the serving of HuggingFace-compatible models at scale. Built on vLLM's Paged Attention technology, it delivers high-performance model inference for multiple concurrent users. Sonar serves as the backend engine powering the [Dolphin Inference Network](https://datagen.dphn.ai) and [PygmalionAI](https://pygmalion.chat)'s chat platforms and API infrastructure.

Sonar builds upon and integrates the exceptional work from [various projects](#acknowledgements), primarily [vLLM](https://vllm.ai).

## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](https://vllm.ai) from vLLM
- Optimized CUDA kernels for improved inference
- Quantization support via [AQLM](https://arxiv.org/abs/2401.06118), [AutoRound](https://arxiv.org/abs/2309.05516), [AWQ](https://arxiv.org/abs/2306.00978), [BitNet](https://arxiv.org/abs/2310.11453), [Bitsandbytes](https://arxiv.org/abs/2208.07339), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [GGUF](https://github.com/ggml-org/llama.cpp), [GPTQ](https://arxiv.org/abs/2210.17323), [QuIP#](https://arxiv.org/abs/2402.04396), [SqueezeLLM](https://arxiv.org/abs/2306.07629), [Marlin](https://arxiv.org/abs/2408.11743), [[2]](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) [[3]](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer), [TorchAO](https://github.com/pytorch/ao), [VPTQ](https://arxiv.org/abs/2409.17066), [compressed_tensors](https://github.com/vllm-project/llm-compressor), [MXFP4](https://huggingface.co/blog/RakshitAralimatti/learn-ai-with-me), and more.
- Distributed inference
- Quantized KV cache using scaled and scale-less FP8, and TurboQuant
- Support for modern samplers such as DRY, XTC, Mirostat, and more
- Disaggregated inference
- Speculative decoding, including EAGLE, DFlash, ngram, MTP, and more
- Multimodal support
- Multi-LoRA support

## Quickstart

Install the engine (the Python package and CLI keep the historical `aphrodite` name for now):

```sh
pip install -U aphrodite-engine
```

Then launch a model:

```sh
aphrodite run Qwen/Qwen3.5-0.8B
```

This will create a [OpenAI](https://platform.openai.com/docs/api-reference/)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports OpenAI, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).

## Requirements

- Operating System: Linux, Windows (WSL2)
- Python: 3.10 to 3.13 (build from source for 3.14)

#### Build Requirements

- CUDA >= 12

### Notes

1. By design, Sonar takes up 92% of your GPU's VRAM. If you're not serving an LLM at scale, you may want to limit the amount of memory it takes up. You can do this in the API example by launching the server with the `--gpu-memory-utilization 0.6` (0.6 means 60%).

2. You can view the full list of commands by running `aphrodite run --help`.

## Acknowledgements

Sonar would have not been possible without the phenomenal work of other open-source projects. A (non-exhaustive) list:

- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [xFormers](https://github.com/facebookresearch/xformers)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM/)
- [Exllamav2](https://github.com/turboderp/exllamav2)
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI)
- [AQLM](https://github.com/Vahe1994/AQLM)
- [KoboldAI](https://github.com/henk717/KoboldAI)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Ray](https://github.com/ray-project/ray)

### Sponsors

Past and present, in alphabetical order:

- [Arc Compute](https://www.arccompute.io/)
- [Lium](https://lium.io)
- [Prime Intellect](https://www.primeintellect.ai/)
- [PygmalionAI](https://pygmalion.chat)
- [Ruliad AI](https://ruliad.ai)

## Contributing

Everyone is welcome to contribute. You can support the project by opening Pull Requests for new features, fixes, or general UX improvements.
