## GPTQ Conversion to Marlin

First, you will need a GPTQ model that satisfies the following conditions:

### Acquiring a compatible GPTQ model

- `group_size=-1` OR `128`
- `bits=4`
- `desc_act=False`

If your model does not meet the requirements above, then run the following script to convert an FP16 model to the appropriate GPTQ format:

```py
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "mistralai/Mistral-7B-Instruct-v0.2"
quantized_model_dir = "/path/to/output"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)

model.save_quantized(quantized_model_dir, use_safetensors=True)
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")
```

Replace the `pretrained_model_dir` and `quantized_model_dir` with the appropriate paths to your base model and output directory. Save the script, and run it like this:

```sh
CUDA_VISIBLE_DEVICES=0 python quantize.py
```

You may need to install the AutoGPTQ library via `pip install auto-gptq`.

Once you have your compatible GPTQ model, follow the steps below to convert it to Marlin format.

### Converting GPTQ models to Marlin

You will need to clone and install the Marlin repository:

```sh
git clone https://github.com/IST-DASLab/marlin && cd marlin

pip install -e .
```

Then simply run the following in this directory:

```sh
python convert.py --model-id /path/to/gptq/model --save-path /path/to/output/marlin
```

That should be all you'll need to do. Then simply launch Sonar, point `--model` to the marlin checkpoint, and that will be all. Happy prompting.
