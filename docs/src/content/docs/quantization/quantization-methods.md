---
title: Quantization Methods
---

Aphrodite supports many different quantization methods. Here we provide an overview of each, along with how to quantize a model using that method. The methods are listed in alphabetical order.

## AQLM

Reference:

- Paper: [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/pdf/2401.06118.pdf)
- Code: [GitHub](https://github.com/Vahe1994/AQLM)

AQLM is a 2-bit quantization method that allows extreme compression of LLMs. It extends [Additive Quantization](https://openaccess.thecvf.com/content_cvpr_2014/papers/Babenko_Additive_Quantization_for_2014_CVPR_paper.pdf) to the task of compressing LLM weights such that the output of each layer and the Transformer block are approximately preserved. It adds two new innovations: (1) adapting the MAP-MRF optimization problem behind AQ to be instance-aware, taking layer calibration input & output activations into accounts; (2) complementing the layer-wise optimization with an efficient intra-block tuning technique, which optimizes quantization parameters jointly over several layers, using only the calibration data.

Producing an AQLM quant is prohibitively expensive, as you need to train and quantize the model at the same time. Quantization of a 70B parameter model to 2-bits takes about 2 weeks on 8xA100 GPUs.

To quantize a model to AQLM, follow these teps:

1. Clone the AQLM repo:

```sh
git clone --recursive https://github.com/Vahe1994/AQLM && cd AQLM
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export MODEL_PATH=<PATH_TO_MODEL_ON_HUB>
export DATASET_PATH=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
export SAVE_PATH=/path/to/save/quantized/model/
export WANDB_PROJECT=MY_AQ_EXPS
export WANDB_NAME=COOL_EXP_NAME

python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1024 \
 --val_size=128 \
 --num_codebooks=1 \
 --nbits_per_codebook=16 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --offload_activations \
 --wandb \
 --resume \
 --save $SAVE_PATH
```

You can then load the quantized model for inference using Aphrodite:

```sh
aphrodite run $SAVE_PATH
```

## AWQ

Reference:

- Paper: [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- Code: [GitHub](https://github.com/mit-han-lab/llm-awq) & [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

AWQ is a quantization method to store the model weights in 4-bit. It achieves this by performing Activation-aware Weight Quantization (AWQ). the method is based on the observation that *weights are not equally important* for LLMs' performance. There's a small fraction (0.1%-1%) of *salient* weights; skipping the quantization of these salient weights will significantly reduce the quantization loss. To find the salient weight channels, the insight is that we should refer to the activation distribution instead of the weight distribution, despite that we are doing *weight-only* quantization: weight channels corresponding to larger activation magnitudes are more salient since they process more important features.

To quantize a model to AWQ, follow these steps:

1. Install Transformers (already installed with Aphrodite):

```sh
pip install transformers
```

Quantize the model:

```py
from transformers import AutoModelForCausalLM, AwqConfig, AutoTokenizer

model_id = "/path/to/model"  # can also be a HF model
tokenizer = AutoTokenizer.from_pretrained(model_id)
awq_config = AwqConfig(
    bits=4,
    dataset="wikitext2",
    group_size=128,
    desc_act=True,
    use_cuda_fp16=True,
    tokenizer=tokenizer
)

model = AutoModeForCausalLM.from_pretrained(model_id, quantization_config=awq_config, attn_implementation="flash_attention_2")
model.config.quantization_config.dataset = None
model.save_pretrained(f"{model_id}-AWQ")
```

You can then load the quantized model for inference using Aphrodite:

```sh
aphrodite run /path/to/model-AWQ
```

:::tip
By default, Aphrodite will load AWQ models using the Marlin kernels for high throughput. If this is undesirable, you can use the `-q awq` flag to load the model using the AWQ library instead.
:::

## BitsAndBytes

Reference:

- [GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes)

BitsAndBytes is a method for runtime quantization of FP16 models.

To get started, simply load an FP16 model with these arguments:

```sh
aphrodite run <model> -q bitsandbytes --load-format bitsandbytes
```

:::warning
Currently, Tensor Parallel does not work with BitsAndBytes quantization.
:::

## DeepspeedFP

Reference:

- [GitHub](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fp6/03-05-2024)

Aphrodite supports weights quantization at runtime using DeepspeedFP. Deepspeed supports Floating-Point quantization to FP4, FP6, FP8, and FP12. To quantize a model using DeepspeedFP, follow these steps:

1. Install Deepspeed:

```sh
pip install deepspeed>=0.14.2
```

1. Load an FP16 model with Aphrodite:

```sh
aphrodite run <model> -q deepspeedfp --deepspeed-fp-bits 6  # or 4, 8, 12
```

## EETQ

Reference:

- [GitHub](https://github.com/NetEase-FuXi/EETQ)

EETQ is an "Easy and Efficient" Quantization method for Transformers. It supports INT8 weight-only quantization.

To quantize a model using EETQ, follow these steps:

```sh
git clone https://github.com/NetEase-FuXi/EETQ.git && cd EETQ
git submodule update --init --recursive
pip install -e .  # this may take a while
```

Quantize the model:

```py
from transformers import AutoModelForCausalLM, EetqConfig
path = "/path/to/model"
quantization_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", quantization_config=quantization_config)
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```

Then, you can load the quantized model for inference using Aphrodite:

```sh
aphrodite run /path/to/quantized/model
```

## FBGEMM_FP8

Reference:

- [GitHub](https://github.com/pytorch/FBGEMM)

Aphrodite supports the FB (Facebook) GEMM (General Matrix Multiply) quantization method for FP8 quantization.

You can use this method to run the official Meta-Llama FP8 models, such as [meta-llama/Meta-Llama-3.1-405B-Instruct-FP8](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8).

To load a model with FBGEMM_FP8 quantization, follow these steps:

```sh
aphrodite run <model>
```

## FP8

Reference:

- [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__FP8.html)
- [Marlin](https://github.com/IST-DASLab/marlin)
Aphrodite supports runtime quantization of LLMs from FP16 to FP8. This method will either use the hardware support present in NVIDIA GPUs (if Ada Lovelace or higher), or will use Marlin kernels for older GPUs (Ampere).

On XPU, non-block FP8 scaled-mm linear layers default to W8A16; setting `--linear-backend xpu` forces W8A8. Use `--linear-backend xpu_woq` to explicitly select weight-only quantization (W8A16).

To load a model with FP8 quantization, follow these steps:

```sh
aphrodite run <model> -q fp8
```

## GGUF

Aphrodite supports loading models serialized in GGUF format, from the popular [llama.cpp](https://github.com/ggerganov/llama.cpp) library. Note that GGUF models are stored as single files instead of directories, so you will need to download the files to disk first, then load them with Aphrodite.

To load a GGUF model, follow these steps:

```sh
aphrodite run /path/to/model.gguf
```

Please refer to the [llama.cpp](https://github.com/ggerganov/llama.cpp) documentation for more information on how to generate GGUF models.

## GPTQ

Reference:

- [GitHub](https://github.com/IST-DASLab/gptq)
- [Paper](https://arxiv.org/abs/2210.17323)

GPTQ is a quantization method for compressing models to 2, 3, 4, and 8 bits. The most commonly used sizes are 4 and 8, as the 2 and 3-bit quants lead to significant accuracy loss.

You can quantize a model to GPTQ using the following steps:

```py
from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer

model_id = "/path/to/model"  # can also be a HF model
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(
    bits=4,
    dataset="wikitext2",
    group_size=128,
    desc_act=True,
    use_cuda_fp16=True,
    tokenizer=tokenizer
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config, attn_implementation="sdpa")
model.config.quantization_config.dataset = None
model.save_pretrained(f"{model_id}-GPTQ")
```

You can then load the quantized model for inference using Aphrodite:

```sh
aphrodite run /path/to/model-GPTQ
```

:::tip
By default, Aphrodite will load GPTQ models using the Marlin kernels for high throughput. If this is undesirable, you can use the `-q gptq` flag to load the model using the GPTQ library instead.
:::

## INT8 W8A8 (LLM-Compressor)

Reference:

- [GitHub](https://github.com/vllm-project/llm-compressor/)

Aphrodite supports LLM Compressor-produced quants. Please refer to their repo on how to generate these quants.

## Quant-LLM

Reference:

- [GitHub](https://github.com/usyd-fsalab/fp6_llm)
- [Paper](https://arxiv.org/abs/2401.14112)

Aphrodite supports loading FP16 models quantized to FP2, FP3, FP4, FP5, FP6, and FP7 using the Quant-LLM method at runtime, to achieve extremely high throughput.

To load a model with Quant-LLM quantization, you can simply run:

```sh
aphrodite run <fp16 model> -q fpX
```

Where `X` is the desired weight quantization: 2, 3, 4, 5, 6, or 7 (although 2 and 3-bit quantization is not recommended due to significant accuracy loss).

We also provide fine-grained control over the exact exponent-mantissa combination, if the user wants to the experiment with other formats:

```sh
aphrodite run <fp16 model> -q quant_llm --quant-llm-exp-bits 4 
```

The valid values for `--quant-llm-exp-bits` are 1, 2, 3, 4, and 5. The heuristic we use to determine mantissa is `weight_bits - exp_bits - 1`, so make sure your provided value does not result in a negative mantissa.

See [here](https://gist.github.com/AlpinDale/17babab5be16f522d4d3b134e171001a) for a list of all valid combinations.

To see accuracy and throughput benchmarks, see [here](https://github.com/PygmalionAI/aphrodite-engine/pull/755).

## The other methods

Aphrodite also supports the following quantization methods:

- [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)
- [QQQ](https://github.com/HandH1998/QQQ)
- [SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM)

Please refer to the respective repositories for more information on how to quantize models using these methods.
