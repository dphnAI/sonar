---
title: Speculative decoding
description: Select, configure, and tune speculative decoding in Sonar.
---

Speculative decoding uses a low-cost proposer to draft tokens. The target model
checks the draft in one forward pass and accepts a valid prefix. Rejected tokens
do not change the output distribution, apart from normal numerical differences
between model runs.

Speculation is most useful for memory-bound decode workloads at low or medium
request rates. It can reduce inter-token latency when the target model accepts
several draft tokens per step. The proposer also uses compute and memory. At
high request rates, this overhead can reduce throughput.

## Supported methods

Sonar accepts these values in `speculative_config.method`.

| Method | Proposer | Extra checkpoint | Typical use |
| --- | --- | :---: | --- |
| `mtp` | Native multi-token prediction layers | Sometimes | Models that include MTP layers or have an assistant checkpoint |
| `dspark` | Parallel block model with a Markov head | Usually | Low-latency block drafting with a trained DSpark checkpoint |
| `dflash` | Parallel diffusion-style block model | Yes | Draft a complete token block in one model pass |
| `eagle`, `eagle3` | Autoregressive EAGLE head | Yes | General model-based speculation |
| `draft_model` | Smaller causal language model | Yes | Target and draft models with compatible vocabularies |
| `ngram` | CPU prompt lookup | No | Repetitive prompts, code, and structured text |
| `ngram_gpu` | GPU prompt lookup | No | Prompt lookup when CPU lookup is a bottleneck |
| `suffix` | GPU prompt and response suffix matching | No | Repeated continuations across requests |
| `extract_hidden_states` | Hidden-state extraction backend | Depends | Specialized trained proposers |

Several old model-family method names remain accepted for compatibility. Sonar
maps names such as `deepseek_mtp`, `qwen3_next_mtp`, and `mimo_mtp` to `mtp`.
Use `method: "mtp"` in new configurations.

The current [supported-model table](/reference/models/) lists registered draft
architectures. A registered architecture does not guarantee that every target
and draft checkpoint pair is compatible.

## Choose a method

Use this order for an initial test:

1. Use MTP when the target checkpoint has supported native MTP layers.
2. Use DSpark or DFlash when a trained checkpoint exists for the target.
3. Use EAGLE-3 when a compatible speculator is available.
4. Use n-gram or suffix decoding for repetitive workloads without a draft
   checkpoint.
5. Use a small draft model when the other methods do not apply.

Start with a small speculative depth. Increase it only when the later positions
have a useful acceptance rate. A large draft block wastes proposer and target
work when most tokens are rejected.

## Pass the configuration

The server accepts a JSON object:

```bash
sonar serve MODEL \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4
  }'
```

Quote the complete object in a shell. Without the outer quotes, the shell
removes the JSON quotes or splits the value. The command then fails during
argument conversion.

The Python API accepts a dictionary:

```python
from sonar import LLM

llm = LLM(
    model="MODEL",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 4,
    },
)
```

`num_speculative_tokens` sets the maximum number of draft tokens per step. Some
methods impose an additional limit from the draft checkpoint.

## MTP tutorial

MTP uses prediction layers trained with the target model. It often needs less
extra memory than a separate draft language model. Sonar shares the target
embedding and language-model head when the architecture supports sharing.

### Use MTP from the target checkpoint

Use this form when the target checkpoint contains its MTP weights:

```bash
sonar serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": 3
  }'
```

When `model` is absent, Sonar loads the MTP layers from the target checkpoint.
It also uses the target quantization setting for those layers.

Use the same configuration with the Python API:

```python
from sonar import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 3,
    },
)

outputs = llm.generate(
    ["Explain continuous batching in two paragraphs."],
    SamplingParams(temperature=0.0, max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

### Use a separate MTP assistant

Some model families publish the prediction layers as an assistant checkpoint.
Pass that checkpoint in `model`:

```bash
sonar serve google/gemma-4-E4B-it \
  --speculative-config '{
    "method": "mtp",
    "model": "google/gemma-4-E4B-it-assistant",
    "num_speculative_tokens": 2
  }'
```

The `model` field names the assistant here. It does not make this a generic
draft-model configuration.

### Tune MTP

Start with one to three speculative tokens. Some checkpoints contain one MTP
layer and reuse it for deeper speculation. Repeated use can lower acceptance at
later positions. Sonar warns about this case.

Check these items when MTP fails during startup:

- Confirm that the target architecture has a registered MTP implementation.
- Confirm that the checkpoint contains the expected MTP weights.
- Reduce `num_speculative_tokens` to the checkpoint prediction depth.
- Set `model` when the model family uses a separate assistant checkpoint.

The [vLLM MTP guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
describes the upstream serving interface. The
[Speculators MTP guide](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/)
explains training and checkpoint conversion.

## DSpark tutorial

DSpark drafts a token block with a parallel backbone. It then applies a
lightweight Markov head from left to right. The Markov head conditions each
sampled token on the preceding sampled token. This design adds dependencies
inside the block without another full transformer pass.

DSpark checkpoints define a block size. Set `num_speculative_tokens` to at least
that value. Sonar rejects a smaller value because it gives the draft model an
unsupported layout. Common checkpoints use a block size of seven.

### Serve Qwen3 with DSpark

This configuration matches Sonar's Qwen3 DSpark end-to-end test:

```bash
sonar serve Qwen/Qwen3-4B-FP8 \
  --trust-remote-code \
  --max-model-len 4096 \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_4b_block7",
    "num_speculative_tokens": 7,
    "attention_backend": "FLASH_ATTN",
    "draft_sample_method": "probabilistic"
  }'
```

`attention_backend` applies to the draft model. DSpark can require an attention
backend that supports its non-causal draft attention. `FLASH_ATTN` is the tested
choice for this Qwen3 checkpoint.

`draft_sample_method: "probabilistic"` samples from the draft distribution and
retains the logits needed by rejection sampling. It uses more GPU memory than
greedy drafting. Remove this field to use the default greedy mode.

### Serve Gemma 4 with DSpark

```bash
sonar serve google/gemma-4-12B-it \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_gemma4_12b_block7",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic"
  }'
```

This model pair needs substantially more memory than the Qwen3 example. Reduce
`--max-num-seqs` before you reduce the DSpark block length. The draft checkpoint
requires the complete block layout.

### DSpark checkpoints embedded in a target

Some targets can contain their DSpark weights. Omit `model` in that case:

```bash
sonar serve TARGET_WITH_DSPARK_WEIGHTS \
  --speculative-config '{
    "method": "dspark",
    "num_speculative_tokens": 7
  }'
```

Sonar then uses the target checkpoint and inherits its quantization setting.

DSpark does not support pipeline parallelism. Use tensor parallelism or data
parallelism for a DSpark deployment.

The upstream
[DSpark runtime reference](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu/spec_decode/dspark/speculator/)
describes the block layout and Markov sampling path.

## DFlash tutorial

DFlash uses a small diffusion-style draft model. It places mask tokens after an
anchor and predicts a block in one forward pass. A non-causal DFlash checkpoint
lets draft positions attend within the complete query block.

### Serve Qwen3 with DFlash

This configuration follows Sonar's acceptance-rate test:

```bash
sonar serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{
    "method": "dflash",
    "model": "z-lab/Qwen3-8B-DFlash-b16",
    "num_speculative_tokens": 16,
    "max_model_len": 32768
  }'
```

The draft `max_model_len` cannot exceed either checkpoint's limit. It can be
smaller than the target limit when you want Sonar to stop speculation for long
sequences.

DFlash selects parallel drafting automatically. Do not set
`parallel_drafting` for this method. Sonar reads the mask token, target hidden
state layers, attention mode, and anchor sampling mode from the draft config.

### Use DFlash from Python

```python
from sonar import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    trust_remote_code=True,
    max_model_len=32768,
    speculative_config={
        "method": "dflash",
        "model": "z-lab/Qwen3-8B-DFlash-b16",
        "num_speculative_tokens": 16,
        "max_model_len": 32768,
    },
)

outputs = llm.generate(
    ["Write a Python function that merges two sorted iterators."],
    SamplingParams(temperature=0.0, max_tokens=512),
)
print(outputs[0].outputs[0].text)
```

If startup reports that non-causal attention is unsupported, set a compatible
draft attention backend:

```json
{
  "method": "dflash",
  "model": "DRAFT_MODEL",
  "num_speculative_tokens": 8,
  "attention_backend": "FLASH_ATTN"
}
```

The
[Speculators DFlash guide](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
explains its block prediction and anchor modes. Use the checkpoint model card
as the source for its supported verifier, block size, and attention mode.

## EAGLE and EAGLE-3

EAGLE uses target hidden states and a trained draft head. EAGLE-3 can consume
multiple target hidden-state layers.

```bash
sonar serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
    "num_speculative_tokens": 3
  }'
```

The target and draft checkpoints must be a trained pair. Similar model names do
not establish compatibility.

## Generic draft model

A smaller causal model can draft for a larger target:

```bash
sonar serve Qwen/Qwen3-4B-Thinking-2507 \
  --speculative-config '{
    "method": "draft_model",
    "model": "Qwen/Qwen3-0.6B",
    "num_speculative_tokens": 5
  }'
```

Set `draft_tensor_parallel_size` to `1` or to the target tensor-parallel size.
The draft model uses one rank by default when supported.

Target and draft models need compatible vocabularies.

With pipeline parallelism, the complete draft model runs on the final target
stage. `SONAR_PP_LAYER_PARTITION` applies only to the target. EAGLE3, DFlash,
and DSpark require a target architecture that supports passing auxiliary hidden
states between stages; unsupported pairs fail at startup.

Set `draft_decode_context_parallel_size` to `1` or the target's decode-context
parallel size. DFlash and DSpark default to `1` unless the draft is full-attention
MLA. `parallel_drafting` is deprecated and has no effect.

## N-gram and suffix decoding

N-gram lookup copies a continuation that already follows a matching token
sequence in the prompt. It needs no model checkpoint.

```bash
sonar serve MODEL \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

Suffix decoding matches each request's prompt and output and a GPU corpus of
previous responses. It supports async scheduling and needs no extra package:

```bash
sonar serve MODEL \
  --speculative-config '{
    "method": "suffix",
    "num_speculative_tokens": 8
  }'
```

Each step drafts exactly `num_speculative_tokens` tokens. Unmatched slots repeat
the last sampled token. Keep the depth moderate unless adaptive verification is
enabled. `suffix_decoding_max_tree_depth` limits suffix-match length.
`suffix_decoding_max_spec_factor` and `suffix_decoding_min_token_prob` are
deprecated and have no effect.

The corpus holds up to `suffix_decoding_corpus_tokens` tokens (default
`1048576`). It is shared across requests, including different cache salts and
LoRA adapters. Set this value to `0` when requests must not influence each
other's decode timing. Setting `suffix_decoding_max_cached_requests` to `0`
also disables the corpus.

For reproducible seeded sampling, disable the corpus and adaptive verification.

### Adaptive verification

Set `enable_adaptive_verification: true` in the speculative configuration to
verify only draft slots whose estimated acceptance justifies their cost. It
supports model-based proposers and suffix decoding. It requires full CUDA
graphs and a backend that replays variable-length decode batches, such as
`TRITON_ATTN` or FlashAttention 3.

## Measure the result

Do not compare only output tokens per second. Record these values with
speculation disabled and enabled:

- Time to first token
- Inter-token latency
- End-to-end request latency
- Request throughput at the expected concurrency
- GPU memory use
- Mean acceptance length
- Per-position acceptance rate

Sonar logs speculative decoding statistics when request statistics are enabled.
Look for `SpecDecoding metrics`. The log includes mean acceptance length,
accepted throughput, drafted throughput, and per-position acceptance rates.

The Prometheus counters use the `sonar:spec_decode_*` prefix. Calculate the
overall acceptance rate with:

```text
rate(sonar:spec_decode_num_accepted_tokens_total[5m])
/
rate(sonar:spec_decode_num_draft_tokens_total[5m])
```

Calculate mean acceptance length, including the target bonus token, with:

```text
1 + (
  rate(sonar:spec_decode_num_accepted_tokens_total[5m])
  /
  rate(sonar:spec_decode_num_drafts[5m])
)
```

Increase the draft length only while later positions have useful acceptance.
Test the expected sampling parameters because temperature and prompt type can
change acceptance.

## Troubleshoot

### The JSON value does not parse

Put single quotes around the complete JSON object in Bash. JSON keys and string
values must retain their double quotes.

### The draft architecture is unsupported

Check the draft checkpoint `architectures` and `model_type` fields. Compare the
architecture with the [supported-model table](/reference/models/). Update Sonar
when the checkpoint needs a newer implementation.

### Startup runs out of memory

Reduce `--max-num-seqs` first. Then reduce the context length or
`--gpu-memory-utilization`. Probabilistic draft sampling retains draft logits
and uses additional memory.

### Speculation makes the service slower

Reduce `num_speculative_tokens`. Check the per-position acceptance rates. Test
at the production concurrency. Disable speculation when proposer work costs
more than the accepted target-model work.

### Long requests stop speculating

The draft model can have a shorter context limit than the target. Increase
`speculative_config.max_model_len` only when the draft checkpoint supports that
length.

## Further reading

- [vLLM speculative decoding guide](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [vLLM Speculators algorithm guide](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/)
- [DFlash paper](https://arxiv.org/abs/2602.06036)
