---
title: Supported Models
---

Aphrodite supports a large variety of generative Transformer models in [Hugging Face Transformers](https://huggingface.co/models). The following is the list of model *architectures* that we currently support.

## Decoder-only Language Models

| Architecture                      |                       Example HF Model |
| --------------------------------- | -------------------------------------: |
| `AquilaForCausalLM`               |                   `BAAI/AquilaChat-7B` |
| `ArcticForCausalLM`               |  `Snowflake/snowflake-arctic-instruct` |
| `BaiChuanForCausalLM`             |      `baichuan-inc/Baichuan2-13B-Chat` |
| `BloomForCausalLM`                |                    `bigscience/bloomz` |
| `ChatGLMModel`                    |                    `THUDM/chatglm3-6b` |
| `CohereForCausalLM`               |       `CohereForAI/c4ai-command-r-v01` |
| `DbrxForCausalLM`                 |             `databricks/dbrx-instruct` |
| `DeciLMForCausalLM`               |                     `DeciLM/DeciLM-7B` |
| `DeepseekForCausalLM`             |    `deepseek-ai/deepseek-moe-16b-base` |
| `DeepseekV2ForCausalLM`           |            `deepseek-ai/DeepSeek-V2.5` |
| `DeepseekV32ForCausalLM`          |            `deepseek-ai/DeepSeek-V3.2` |
| `ExaoneForCausalLM`               | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` |
| `FalconForCausalLM`               |                     `tiiuae/falcon-7b` |
| `GPT2LMHeadModel`                 |                                 `gpt2` |
| `GPTBigCodeForCausalLM`           |                    `bigcode/starcoder` |
| `GPTJForCausalLM`                 |             `pygmalionai/pygmalion-6b` |
| `GPTNeoXForCausalLM`              |                `EleutherAI/pythia-12b` |
| `GemmaForCausalLM`                |                      `google/gemma-7b` |
| `Gemma2ForCausalLM`               |                    `google/gemma-2-9b` |
| `GraniteForCausalLM`              |              `ibm-research/PowerLM-3b` |
| `GraniteMoeForCausalLM`           |             `ibm-research/PowerMoE-3b` |
| `InternLMForCausalLM`             |                 `internlm/internlm-7b` |
| `InternLM2ForCausalLM`            |                `internlm/internlm2-7b` |
| `JAISLMHeadModel`                 |                      `core42/jais-13b` |
| `JambaForCausalLM`                |                  `ai21labs/Jamba-v0.1` |
| `LlamaForCausalLM`                |         `meta-llama/Meta-Llama-3.1-8B` |
| `MPTForCausalLM`                  |                      `mosaicml/mpt-7b` |
| `MambaForCausalLM`                |           `state-spaces/mamba-2.8b-hf` |
| `MiniCPMForCausalLM`              |          `openbmb/MiniCPM-2B-dpo-bf16` |
| `MiniCPM3ForCausalLM`             |                  `openbmb/MiniCPM3-4B` |
| `MistralForCausalLM`              |            `mistralai/Mistral-7B-v0.1` |
| `MixtralForCausalLM`              |          `mistralai/Mixtral-8x7B-v0.1` |
| `NemotronForCausalLM`             |              `nvidia/Minitron-8B-Base` |
| `NVLM_D`                          |                    `nvidia/NVLM-D-72B` |
| `OPTForCausalLM`                  |                     `facebook/opt-66b` |
| `OlmoForCausalLM`                 |                   `allenai/OLMo-7B-hf` |
| `Olmo2ForCausalLM`                |               `allenai/OLMo-2-0425-1B` |
| `OlmoeForCausalLM`                |             `allenai/OLMoE-1B-7B-0125` |
| `OrionForCausalLM`                |           `OrionStarAI/Orion-14B-Chat` |
| `PhiForCausalLM`                  |                      `microsoft/phi-2` |
| `Phi3ForCausalLM`                 | `microsoft/Phi-3-medium-128k-instruct` |
| `Phi3SmallForCausalLM`            |  `microsoft/Phi-3-small-128k-instruct` |
| `PhiMoEForCausalLM`               |       `microsoft/Phi-3.5-MoE-instruct` |
| `QwenLMHeadModel`                 |                         `Qwen/Qwen-7B` |
| `Qwen2ForCausalLM`                |                       `Qwen/Qwen2-72B` |
| `Qwen2MoeForCausalLM`             |               `Qwen/Qwen1.5-MoE-A2.7B` |
| `Qwen2VLForConditionalGeneration` |            `Qwen/Qwen2-VL-7B-Instruct` |
| `SolarForCausalLM`                |   `upstage/solar-pro-preview-instruct` |
| `StableLmforCausalLM`             |         `stabilityai/stablelm-3b-4e1t` |
| `Starcoder2ForCausalLM`           |                `bigcode/starcoder2-3b` |
| `XverseForCausalLM`               |               `xverse/XVERSE-65B-Chat` |

:::tip
On ROCm platforms, Mistral and Mixtral are capped to 4096 max context length due to sliding window issues.
:::

## Encoder-Decoder Language Models

| Architecture                   |             Example Model |
| ------------------------------ | ------------------------: |
| `BartForConditionalGeneration` | `facebook/bart-large-cnn` |

## Embedding Models

| Architecture          | Example Model                     |
| --------------------- | --------------------------------- |
| `MistralModel`        | `intfloat/e5-mistral-7b-instruct` |
| `Qwen2ForRewardModel` | `Qwen/Qwen2.5-Math-RM-72B`        |
| `Gemma2Model`         | `BAAI/bge-multilingual-gemma2`    |

### Pooling Configuration Resolution

For pooling models, the pooling method and `use_activation` are resolved per
field. An explicitly set field in `--pooler-config` takes precedence over
Sentence Transformers metadata, which in turn takes precedence over the model
architecture or task default. Fields left unset continue through the chain
independently.

The current `PoolerConfig` has no `normalize` or `activation` field.
`use_activation` controls whether the task's constructed normalization or
classification activation is applied.

| Field | Source precedence | How to override |
| ----- | ----------------- | --------------- |
| Pooling method (`pooling_type`) | `--pooler-config` > boolean `pooling_mode_*` fields in the Pooling module referenced by Sentence Transformers `modules.json` > architecture default (`LAST` for sequence pooling and `ALL` for token pooling unless the architecture overrides it) | Set `{"pooling_type": "CLS"}`, or set `seq_pooling_type` / `tok_pooling_type` explicitly. |
| Embedding normalization (`use_activation`) | `--pooler-config` > Sentence Transformers modules (`true` when a Normalize module is present, otherwise `false`) > pooling-task default (`true`) when no Sentence Transformers Pooling module is found | Set `{"use_activation": false}` to return unnormalized embeddings. |
| Classification activation function | Hugging Face `problem_type` > Sentence Transformers activation metadata > sigmoid or softmax selected from the label count | The function cannot be selected through `--pooler-config`; set `{"use_activation": false}` to return logits instead. |

Sentence Transformers configurations using the newer compact `pooling_mode`
string are not currently parsed; see
[vLLM issue #45995](https://github.com/vllm-project/vllm/issues/45995).

For converted models and predefined models using the standard DispatchPooler
adapters, `embed` and `token_embed` construct an L2-normalization head, while
`classify` and `token_classify` construct the selected classification activation.
In both cases, `use_activation` controls whether that head is applied. Models
with custom poolers can implement different behavior.

To inspect the resolved fields without loading model weights:

```python
from aphrodite.config import ModelConfig, PoolerConfig
from aphrodite.model_executor.layers.pooler.activations import get_act_fn


def inspect(requested: PoolerConfig) -> None:
    model_config = ModelConfig(
        "intfloat/e5-small",
        runner="pooling",
        pooler_config=requested,
    )
    resolved = model_config.pooler_config
    assert resolved is not None
    print(
        {
            "seq_pooling_type": resolved.seq_pooling_type,
            "tok_pooling_type": resolved.tok_pooling_type,
            "use_activation": resolved.use_activation,
            "sequence_classification_activation": type(
                get_act_fn(model_config.hf_config)
            ).__name__,
        }
    )


inspect(PoolerConfig())
inspect(PoolerConfig(pooling_type="CLS", use_activation=False))
```

For `intfloat/e5-small`, the first result contains `MEAN`, `ALL`, and `True`.
The second contains `CLS`, `ALL`, and `False`. Both report the classification
activation that the standard sequence-classification adapter would construct.

## Multimodal Language Models

| Architecture                             | Supported Modalities |                              Example Model |
| ---------------------------------------- | :------------------: | -----------------------------------------: |
| `Blip2ForConditionalGeneration`          |        Image         |                `Salesforce/blip2-opt-6.7b` |
| `ChameleonForConditionalGeneration`      |        Image         |                    `facebook/chameleon-7b` |
| `ChatGLMModel`                           |        Image         |                        `THUDM/chatglm3-6b` |
| `Cosmos3ForConditionalGeneration`        |    Image, Video      | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super` |
| `Cosmos3EdgeForConditionalGeneration`    |    Image, Video      |                      `nvidia/Cosmos3-Edge` |
| `InternVLChatModel`                      |        Image         |                   `OpenGVLab/InternVL2-8B` |
| `LlavaForConditionalGeneration`          |        Image         |                `llava-hf/llava-v1.5-7b-hf` |
| `LlavaNextForConditionalGeneration`      |        Image         |        `llava-hf/llava-v1.6-mistral-7b-hf` |
| `LlavaNextVideoForConditionalGeneration` |        Video         |          `llava-hf/LLaVA-NeXT-Video-7B-hf` |
| `LlavaOnevision2ForConditionalGeneration` |    Image, Video     | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` |
| `LlavaOnevisionForConditionalGeneration` |     Image, Video     |  `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| `MiniCPMV`                               |        Image         |                    `openbmb/MiniCPM-V-2_6` |
| `MllamaForConditionalGeneration`         |        Image         | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| `MolmoForCausalLM`                       |        Image         |                  `allenai/Molmo-7B-D-0924` |
| `MossTranscribeDiarizeForConditionalGeneration` | Audio | `OpenMOSS-Team/MOSS-Transcribe-Diarize` |
| `PaliGemmaForConditionalGeneration`      |        Image         |               `google/paligemma-3b-pt-224` |
| `Phi3VForCausalLM`                       |        Image         |        `microsoft/Phi-3.5-vision-instruct` |
| `PixtralForConditionalGeneration`        |        Image         |               `mistralai/Pixtral-12B-2409` |
| `QWenLMHeadModel`                        |        Image         |                             `Qwen/Qwen-VL` |
| `Qwen2VLForConditionalGeneration`        |        Image         |                `Qwen/Qwen2-VL-7B-Instruct` |
| `UltravoxModel`                          |        Audio         |                   `fixie-ai/ultravox-v0_3` |

## Speculative Models

| Architecture                   | Example Model                            |
| ------------------------------ | ---------------------------------------- |
| `EAGLEModel`                   | `abhigoyal/aphrodite-eagle-llama-68m-random`  |
| `MedusaModel`                  | `abhigoyal/aphrodite-medusa-llama-68m-random` |
| `MLPSpeculatorPreTrainedModel` | `ibm-fms/llama-160m-accelerator`         |

If your model uses any of the architectures above, you can seamlessly run your model with Aphrodite.
