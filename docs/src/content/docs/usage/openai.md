---
title: OpenAI-Compatible API Server
---

This page assumes you've already installed Aphrodite and know how to launch the OpenAI-Compatible server.

:::tip
This page is quite large and extensive; please use the table of contents ("On this page" to the top left) to navigate.
:::

:::tip[Human-readable integer arguments]
Many integer arguments accept human-readable suffixes for convenience. For example:

- `1k` = 1,000 (decimal kilo)
- `1K` = 1,024 (binary kibibyte)
- `1m` = 1,000,000 (decimal mega)
- `1M` = 1,048,576 (binary mebibyte)
- `1g` / `1G` = 1 billion / 1 gibibyte
- `1t` / `1T` = 1 trillion / 1 tebibyte

Decimal suffixes (`k`, `m`, `g`, `t`) also accept floating point: `25.6k` = 25,600.
Binary suffixes (`K`, `M`, `G`, `T`) require integers: `32K` = 32,768.

Supported arguments include: `--max-model-len`, `--max-num-batched-tokens`, `--max-num-scheduled-tokens`, `--kv-cache-memory-bytes`, `--safetensors-prefetch-block-size`.
:::

## API Reference

Please see  the [OpenAI API Reference](https://platform.openai.com/docs/api-reference) for more information on the API scheme, as we support all parameters, except:

- in `/v1/chat/completions`: `tools` and `tool_choice`
- in `/v1/completions`: `suffix`

Otherwise, we support everything, plus many other parameters.

Aphrodite also provides experimental support for the OpenAI Vision API.

## Extra Parameters

If using the `openai` python library, you cannot pass extra parameters such as `min_p`, `guided_choice`, etc. Thankfully, the library allows you to extend the body as needed:

```py
completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: LLMs are wonderful!"}
    ],
    extra_body={
        "guided_choice": ["positive", "negative"]
    }
)
```

## Extra Parameters for Chat API

Aphrodite supports the following extra parameters that are not supported by OpenAI:

```py
best_of: Optional[int] = None
use_beam_search: Optional[bool] = False
top_k: Optional[int] = -1
min_p: Optional[float] = 0.0
top_a: Optional[float] = 0.0
tfs: Optional[float] = 1.0
eta_cutoff: Optional[float] = 0.0
epsilon_cutoff: Optional[float] = 0.0
typical_p: Optional[float] = 1.0
smoothing_factor: Optional[float] = 0.0
smoothing_curve: Optional[float] = 1.0
repetition_penalty: Optional[float] = 1.0
length_penalty: Optional[float] = 1.0
early_stopping: Optional[bool] = False
ignore_eos: Optional[bool] = False
min_tokens: Optional[int] = 0
stop_token_ids: Optional[List[int]] = Field(default_factory=list)
skip_special_tokens: Optional[bool] = True
spaces_between_special_tokens: Optional[bool] = True
truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
```

And the following parameters:

```py
echo: Optional[bool] = Field(
    default=False,
    description=(
        "If true, the new message will be prepended with the last message "
        "if they belong to the same role."),
)
add_generation_prompt: Optional[bool] = Field(
    default=True,
    description=
    ("If true, the generation prompt will be added to the chat template. "
        "This is a parameter used by chat template in tokenizer config of the "
        "model."),
)
add_special_tokens: Optional[bool] = Field(
    default=False,
    description=(
        "If true, special tokens (e.g. BOS) will be added to the prompt "
        "on top of what is added by the chat template. "
        "For most models, the chat template takes care of adding the "
        "special tokens so this should be set to False (as is the "
        "default)."),
)
documents: Optional[List[Dict[str, str]]] = Field(
    default=None,
    description=
    ("A list of dicts representing documents that will be accessible to "
        "the model if it is performing RAG (retrieval-augmented generation)."
        " If the template does not support RAG, this argument will have no "
        "effect. We recommend that each document should be a dict containing "
        "\"title\" and \"text\" keys."),
)
chat_template: Optional[str] = Field(
    default=None,
    description=(
        "A Jinja template to use for this conversion. "
        "If this is not passed, the model's default chat template will be "
        "used instead."),
)
chat_template_kwargs: Optional[Dict[str, Any]] = Field(
    default=None,
    description=("Additional kwargs to pass to the template renderer. "
                    "Will be accessible by the chat template."),
)
include_stop_str_in_output: Optional[bool] = Field(
    default=False,
    description=(
        "Whether to include the stop string in the output. "
        "This is only applied when the stop or stop_token_ids is set."),
)
guided_json: Optional[Union[str, dict, BaseModel]] = Field(
    default=None,
    description=("If specified, the output will follow the JSON schema."),
)
guided_regex: Optional[str] = Field(
    default=None,
    description=(
        "If specified, the output will follow the regex pattern."),
)
guided_choice: Optional[List[str]] = Field(
    default=None,
    description=(
        "If specified, the output will be exactly one of the choices."),
)
guided_grammar: Optional[str] = Field(
    default=None,
    description=(
        "If specified, the output will follow the context free grammar."),
)
guided_decoding_backend: Optional[str] = Field(
    default=None,
    description=(
        "If specified, will override the default guided decoding backend "
        "of the server for this specific request. If set, must be either "
        "'outlines' / 'lm-format-enforcer'"))
guided_whitespace_pattern: Optional[str] = Field(
    default=None,
    description=(
        "If specified, will override the default whitespace pattern "
        "for guided json decoding."))
```

## Extra Parameters for Text Completions API

Aphrodite supports the following extra parameters that are not supported by OpenAI:

```py
use_beam_search: Optional[bool] = False
top_k: Optional[int] = -1
min_p: Optional[float] = 0.0
top_a: Optional[float] = 0.0
tfs: Optional[float] = 1.0
eta_cutoff: Optional[float] = 0.0
epsilon_cutoff: Optional[float] = 0.0
typical_p: Optional[float] = 1.0
smoothing_factor: Optional[float] = 0.0
smoothing_curve: Optional[float] = 1.0
repetition_penalty: Optional[float] = 1.0
length_penalty: Optional[float] = 1.0
early_stopping: Optional[bool] = False
stop_token_ids: Optional[List[int]] = Field(default_factory=list)
ignore_eos: Optional[bool] = False
min_tokens: Optional[int] = 0
skip_special_tokens: Optional[bool] = True
spaces_between_special_tokens: Optional[bool] = True
truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
allowed_token_ids: Optional[List[int]] = None
include_stop_str_in_output: Optional[bool] = False
add_special_tokens: Optional[bool] = False
```

And the following parameters:

```py
response_format: Optional[ResponseFormat] = Field(
    default=None,
    description=
    ("Similar to chat completion, this parameter specifies the format of "
        "output. Only {'type': 'json_object'} or {'type': 'text' } is "
        "supported."),
)
guided_json: Optional[Union[str, dict, BaseModel]] = Field(
    default=None,
    description=("If specified, the output will follow the JSON schema."),
)
guided_regex: Optional[str] = Field(
    default=None,
    description=(
        "If specified, the output will follow the regex pattern."),
)
guided_choice: Optional[List[str]] = Field(
    default=None,
    description=(
        "If specified, the output will be exactly one of the choices."),
)
guided_grammar: Optional[str] = Field(
    default=None,
    description=(
        "If specified, the output will follow the context free grammar."),
)
guided_decoding_backend: Optional[str] = Field(
    default=None,
    description=(
        "If specified, will override the default guided decoding backend "
        "of the server for this specific request. If set, must be one of "
        "'outlines' / 'lm-format-enforcer'"))
guided_whitespace_pattern: Optional[str] = Field(
    default=None,
    description=(
        "If specified, will override the default whitespace pattern "
        "for guided json decoding."))
```

## Chat Template

In order for the LLM to support chat completions protocol, Aphrodite requires the model to include a chat template in its tokenizer config. The chat template is Jinja2 template file that specifies how roles, messages, and other chat-specific tokens are encoded in the input.

Most modern LLMs provide this if they're an Instruct/Chat finetune, but sometimes they may not. For those models, you can manually specify their chat template in the `--chat-template` (or `- chat_template` in the YAML) with the path being the URL or local disk path. You may also provide it as in-line string to the argument. Without a chat template, the server will only launch text completions.

Aphrodite provides a set of chat templates, which you can view [here](https://github.com/PygmalionAI/aphrodite-engine/tree/main/examples/chat_templates).

## Command-line arguments for the server

```console
usage: aphrodite run <model_tag> [options]

positional arguments:
  model_tag             The model tag to serve

options:
  -h, --help            show this help message and exit
  --host HOST           host name
  --port PORT           port number
  --uvicorn-log-level {debug,info,warning,error,critical,trace}
                        log level for uvicorn
  --allow-credentials   allow credentials
  --allowed-origins ALLOWED_ORIGINS
                        allowed origins
  --allowed-methods ALLOWED_METHODS
                        allowed methods
  --allowed-headers ALLOWED_HEADERS
                        allowed headers
  --api-keys API_KEYS   If provided, the server will require this key to be presented in the header.
  --admin-key ADMIN_KEY
                        If provided, the server will require this key to be presented in the header for admin operations.
  --lora-modules LORA_MODULES [LORA_MODULES ...]
                        LoRA module configurations in either 'name=path' formator JSON format. Example (old format): 'name=path' Example (new
                        format): '{"name": "name", "local_path": "path", "base_model_name": "id"}'
  --prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]
                        Prompt adapter configurations in the format name=path. Multiple adapters can be specified.
  --chat-template CHAT_TEMPLATE
                        The file path to the chat template, or the template in single-line form for the specified model
  --response-role RESPONSE_ROLE
                        The role name to return if `request.add_generation_prompt=true`.
  --ssl-keyfile SSL_KEYFILE
                        The file path to the SSL key file
  --ssl-certfile SSL_CERTFILE
                        The file path to the SSL cert file
  --ssl-ca-certs SSL_CA_CERTS
                        The CA certificates file
  --ssl-cert-reqs SSL_CERT_REQS
                        Whether client certificate is required (see stdlib ssl module's)
  --root-path ROOT_PATH
                        FastAPI root_path when app is behind a path based routing proxy
  --middleware MIDDLEWARE
                        Additional ASGI middleware to apply to the app. We accept multiple --middleware arguments. The value should be an
                        import path. If a function is provided, Aphrodite will add it to the server using @app.middleware('http'). If a class
                        is provided, Aphrodite will add it to the server using app.add_middleware().
  --launch-kobold-api   Launch the Kobold API server alongside the OpenAI server
  --max-log-len MAX_LOG_LEN
                        Max number of prompt characters or prompt ID numbers being printed in log. Default: 0
  --return-tokens-as-token-ids
                        When --max-logprobs is specified, represents single tokens asstrings of the form 'token_id:{token_id}' so that tokens
                        thatare not JSON-encodable can be identified.
  --disable-frontend-multiprocessing
                        If specified, will run the OpenAI frontend server in the same process as the model serving engine.
  --allow-inline-model-loading
                        If specified, will allow the model to be switched inline in the same process as the OpenAI frontend server.
  --enable-auto-tool-choice
                        Enable auto tool choice for supported models. Use --tool-call-parserto specify which parser to use
  --tool-call-parser {hermes,internlm,llama3_json,mistral} or name registered in --tool-parser-plugin
                        Select the tool call parser depending on the model that you're using. This is used to parse the model-generated tool
                        call into OpenAI API format. Required for --enable-auto-tool-choice.
  --tool-parser-plugin TOOL_PARSER_PLUGIN
                        Specify the tool parser plugin path to parse model-generated tool calls into OpenAI API format. The parsers registered
                        in this plugin can be referenced in --tool-call-parser.
  --model MODEL         Category: Model Options name or path of the huggingface model to use
  --seed SEED           Category: Model Options random seed
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
                        Category: API Options The model name(s) used in the API. If multiple names are provided, the server will respond to
                        any of the provided names. The model name in the model field of a response will be the first name in this list. If not
                        specified, the model name will be the same as the `--model` argument. Noted that this name(s)will also be used in
                        `model_name` tag content of prometheus metrics, if multiple names provided, metricstag will take the first one.
  --tokenizer TOKENIZER
                        Category: Model Options name or path of the huggingface tokenizer to use
  --revision REVISION   Category: Model Options the specific model version to use. It can be a branch name, a tag name, or a commit id. If
                        unspecified, will use the default version.
  --code-revision CODE_REVISION
                        Category: Model Options the specific revision to use for the model code on Hugging Face Hub. It can be a branch name,
                        a tag name, or a commit id. If unspecified, will use the default version.
  --tokenizer-revision TOKENIZER_REVISION
                        Category: Model Options the specific tokenizer version to use. It can be a branch name, a tag name, or a commit id. If
                        unspecified, will use the default version.
  --tokenizer-mode {auto,slow,mistral}
                        The tokenizer mode. * "auto" will use the fast tokenizer if available. * "slow" will always use the slow tokenizer. *
                        "mistral" will always use the `mistral_common` tokenizer.
  --trust-remote-code   Category: Model Options trust remote code from huggingface
  --download-dir DOWNLOAD_DIR
                        Category: Model Options directory to download and load the weights, default to the default cache dir of huggingface
  --max-model-len MAX_MODEL_LEN
                        Category: Model Options model context length. If unspecified, will be automatically derived from the model.
  --max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE
                        Category: Model Options Maximum context length covered by CUDA graphs. When a sequence has context length larger than
                        this, we fall back to eager mode. (DEPRECATED. Use --max-seq_len-to-capture instead)
  --max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE
                        Maximum sequence length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to
                        eager mode. Additionally for encoder-decoder models, if the sequence length of the encoder input is larger than this,
                        we fall back to the eager mode.
  --rope-scaling ROPE_SCALING
                        Category: Model Options RoPE scaling configuration in JSON format. For example, {"type":"dynamic","factor":2.0}
  --rope-theta ROPE_THETA
                        Category: Model Options RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE theta improves the
                        performance of the scaled model.
  --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
                        Category: Model Options Extra config for model loader. This will be passed to the model loader corresponding to the
                        chosen load_format. This should be a JSON string that will be parsed into a dictionary.
  --enforce-eager [ENFORCE_EAGER]
                        Category: Model Options Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for
                        maximal performance and flexibility.
  --skip-tokenizer-init
                        Category: Model Options Skip initialization of tokenizer and detokenizer
  --tokenizer-pool-size TOKENIZER_POOL_SIZE
                        Category: Model Options Size of tokenizer pool to use for asynchronous tokenization. If 0, will use synchronous
                        tokenization.
  --tokenizer-pool-type TOKENIZER_POOL_TYPE
                        Category: Model Options The type of tokenizer pool to use for asynchronous tokenization. Ignored if
                        tokenizer_pool_size is 0.
  --tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG
                        Category: Model Options Extra config for tokenizer pool. This should be a JSON string that will be parsed into a
                        dictionary. Ignored if tokenizer_pool_size is 0.
  --limit-mm-per-prompt LIMIT_MM_PER_PROMPT
                        For each multimodal plugin, limit how many input instances to allow for each prompt. Expects a comma-separated list of
                        items, e.g.: `image=16,video=2` allows a maximum of 16 images and 2 videos per prompt. Defaults to 1 for each
                        modality.
  --mm-processor-kwargs MM_PROCESSOR_KWARGS
                        Overrides for the multimodal input mapping/processing,e.g., image processor. For example: {"num_crops": 4}.
  --max-logprobs MAX_LOGPROBS
                        Category: Model Options maximum number of log probabilities to return.
  --device {auto,cuda,neuron,cpu,openvino,tpu,xpu}
                        Category: Model Options Device to use for model execution.
  --load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral}
                        Category: Model Options The format of the model weights to load. * "auto" will try to load the weights in the
                        safetensors format and fall back to the pytorch bin format if safetensors format is not available. * "pt" will load
                        the weights in the pytorch bin format. * "safetensors" will load the weights in the safetensors format. * "npcache"
                        will load the weights in pytorch format and store a numpy cache to speed up the loading. * "dummy" will initialize the
                        weights with random values, which is mainly for profiling. * "tensorizer" will load the weights using tensorizer from
                        CoreWeave. See the Tensorize Aphrodite Model script in the Examples section for more information. * "bitsandbytes"
                        will load the weights using bitsandbytes quantization.
  --config-format {auto,hf,mistral}
                        The format of the model config to load. * "auto" will try to load the config in hf format if available else it will
                        try to load in mistral format. Mistral format is specific to mistral models and is not compatible with other models.
  --dtype {auto,half,float16,bfloat16,float,float32}
                        Category: Model Options Data type for model weights and activations. * "auto" will use FP16 precision for FP32 and
                        FP16 models, and BF16 precision for BF16 models. * "half" for FP16. Recommended for AWQ quantization. * "float16" is
                        the same as "half". * "bfloat16" for a balance between precision and range. * "float" is shorthand for FP32 precision.
                        * "float32" for FP32 precision.
  --ignore-patterns IGNORE_PATTERNS
                        Category: Model Options The pattern(s) to ignore when loading the model.Defaults to 'original/**/*' to avoid repeated
                        loading of llama's checkpoints.
  --worker-use-ray      Category: Parallel Options Deprecated, use --distributed-executor-backend=ray.
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Category: Parallel Options number of tensor parallel replicas, i.e. the number of GPUs to use.
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Category: Parallel Options number of pipeline stages. Currently not supported.
  --ray-workers-use-nsight
                        Category: Parallel Options If specified, use nsight to profile ray workers
  --disable-custom-all-reduce
                        Category: Model Options See ParallelConfig
  --distributed-executor-backend {ray,mp}
                        Category: Parallel Options Backend to use for distributed serving. When more than 1 GPU is used, will be automatically
                        set to "ray" if installed or "mp" (multiprocessing) otherwise.
  --max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS
                        Category: Parallel Options load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel
                        and large models
  --quantization {aqlm,awq,deepspeedfp,tpu_int8,eetq,fp8,quant_llm,fbgemm_fp8,modelopt,gguf,marlin,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,quip,squeezellm,compressed-tensors,compressed_tensors,bitsandbytes,qqq,hqq,experts_int8,fp2,fp3,fp4,fp5,fp6,fp7,neuron_quant,vptq,ipex,None}, -q {aqlm,awq,deepspeedfp,tpu_int8,eetq,fp8,quant_llm,fbgemm_fp8,modelopt,gguf,marlin,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,quip,squeezellm,compressed-tensors,compressed_tensors,bitsandbytes,qqq,hqq,experts_int8,fp2,fp3,fp4,fp5,fp6,fp7,neuron_quant,vptq,ipex,None}
                        Category: Quantization Options Method used to quantize the weights. If None, we first check the `quantization_config`
                        attribute in the model config file. If that is None, we assume the model weights are not quantized and use `dtype` to
                        determine the data type of the weights.
  --quantization-param-path QUANTIZATION_PARAM_PATH
                        Category: Quantization Options Path to the JSON file containing the KV cache scaling factors. This should generally be
                        supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling factors default to 1.0, which may cause accuracy
                        issues. FP8_E5M2 (without scaling) is only supported on cuda versiongreater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is
                        instead supported for common inference criteria.
  --preemption-mode PREEMPTION_MODE
                        Category: Scheduler Options If 'recompute', the engine performs preemption by block swapping; If 'swap', the engine
                        performs preemption by block swapping.
  --deepspeed-fp-bits DEEPSPEED_FP_BITS
                        Category: Quantization Options Number of floating bits to use for the deepspeed quantization. Supported bits are: 4,
                        6, 8, 12.
  --quant-llm-fp-bits QUANT_LLM_FP_BITS
                        Category: Quantization Options Number of floating bits to use for the quant_llm quantization. Supported bits are: 4 to
                        15.
  --quant-llm-exp-bits QUANT_LLM_EXP_BITS
                        Category: Quantization Options Number of exponent bits to use for the quant_llm quantization. Supported bits are: 1 to
                        5.
  --kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}
                        Category: Cache Options Data type for kv cache storage. If "auto", will use model data type. CUDA 11.8+ supports fp8
                        (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports fp8 (=fp8_e4m3)
  --block-size {8,16,32}
                        Category: Cache Options token block size for contiguous chunks of tokens. This is ignored on neuron devices and set to
                        max-model-len.
  --enable-prefix-caching, --context-shift
                        Category: Cache Options Enable automatic prefix caching.
  --num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE
                        Category: Cache Options Options If specified, ignore GPU profiling result and use this number of GPU blocks. Used for
                        testing preemption.
  --disable-sliding-window
                        Category: KV Cache Options Disables sliding window, capping to sliding window size
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION, -gmu GPU_MEMORY_UTILIZATION
                        Category: Cache Options The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.If
                        unspecified, will use the default value of 0.9.
  --swap-space SWAP_SPACE
                        Category: Cache Options CPU swap space size (GiB) per GPU
  --cpu-offload-gb CPU_OFFLOAD_GB
                        Category: Cache Options The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading.
                        Intuitively, this argument can be seen as a virtual way to increase the GPU memory size. For example, if you have one
                        24 GB GPU and set this to 10, virtually you can think of it as a 34 GB GPU. Then you can load a 13B model with BF16
                        weight,which requires at least 26GB GPU memory. Note that this requires fast CPU-GPU interconnect, as part of the
                        model isloaded from CPU memory to GPU memory on the fly in each model forward pass.
  --use-v2-block-manager
                        Use BlockSpaceMangerV2. By default this is set to True. Set to False to use BlockSpaceManagerV1
  --scheduler-delay-factor SCHEDULER_DELAY_FACTOR, -sdf SCHEDULER_DELAY_FACTOR
                        Category: Scheduler Options Apply a delay (of delay factor multiplied by previous prompt latency) before scheduling
                        next prompt.
  --enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]
                        Category: Scheduler Options If True, the prefill requests can be chunked based on the max_num_batched_tokens.
  --guided-decoding-backend {outlines,lm-format-enforcer,xgrammar}
                        Which engine will be used for guided decoding (JSON schema / regex etc) by default. Currently support
                        https://github.com/outlines-dev/outlines,https://github.com/mlc-ai/xgrammar, and https://github.com/noamgat/lm-format-
                        enforcer. Can be overridden per request via guided_decoding_backend parameter.
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
                        Category: KV Cache Options maximum number of batched tokens per iteration
  --max-num-seqs MAX_NUM_SEQS
                        Category: API Options maximum number of sequences per iteration
  --single-user-mode    Category: API Options If True, we only allocate blocks for one sequence and use the maximum sequence length as the
                        number of tokens.
  --num-scheduler-steps NUM_SCHEDULER_STEPS
                        Maximum number of forward steps per scheduler call.
  --multi-step-stream-outputs [MULTI_STEP_STREAM_OUTPUTS]
                        If False, then multi-step will stream outputs at the end of all steps
  --num-lookahead-slots NUM_LOOKAHEAD_SLOTS
                        Category: Speculative Decoding Options Experimental scheduling config necessary for speculative decoding. This will be
                        replaced by speculative decoding config in the future; it is present for testing purposes until then.
  --speculative-model SPECULATIVE_MODEL
                        Category: Speculative Decoding Options The name of the draft model to be used in speculative decoding.
  --speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,eetq,fp8,quant_llm,fbgemm_fp8,modelopt,gguf,marlin,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,quip,squeezellm,compressed-tensors,compressed_tensors,bitsandbytes,qqq,hqq,experts_int8,fp2,fp3,fp4,fp5,fp6,fp7,neuron_quant,vptq,ipex,None}
                        Method used to quantize the weights of speculative model.If None, we first check the `quantization_config` attribute
                        in the model config file. If that is None, we assume the model weights are not quantized and use `dtype` to determine
                        the data type of the weights.
  --num-speculative-tokens NUM_SPECULATIVE_TOKENS
                        Category: Speculative Decoding Options The number of speculative tokens to sample from the draft model in speculative
                        decoding
  --speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN
                        Category: Speculative Decoding Options The maximum sequence length supported by the draft model. Sequences over this
                        length will skip speculation.
  --ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX
                        Category: Speculative Decoding Options Max size of window for ngram prompt lookup in speculative decoding.
  --ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN
                        Category: Speculative Decoding Options Min size of window for ngram prompt lookup in speculative decoding.
  --speculative-disable-mqa-scorer
                        If set to True, the MQA scorer will be disabled in speculative and fall back to batch expansion
  --speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE, -spec-draft-tp SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE
                        Category: Speculative Decoding Options Number of tensor parallel replicas for the draft model in speculative decoding.
  --speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE
                        Category: Speculative Decoding Options Disable speculative decoding for new incoming requests if the number of enqueue
                        requests is larger than this value.
  --spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}
                        Category: Speculative Decoding Options Specify the acceptance method to use during draft token verification in
                        speculative decoding. Two types of acceptance routines are supported: 1) RejectionSampler which does not allow
                        changing the acceptance rate of draft tokens, 2) TypicalAcceptanceSampler which is configurable, allowing for a higher
                        acceptance rate at the cost of lower quality, and vice versa.
  --typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD
                        Category: Speculative Decoding Options Set the lower bound threshold for the posterior probability of a token to be
                        accepted. This threshold is used by the TypicalAcceptanceSampler to make sampling decisions during speculative
                        decoding. Defaults to 0.09
  --typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA
                        Category: Speculative Decoding Options A scaling factor for the entropy-based threshold for token acceptance in the
                        TypicalAcceptanceSampler. Typically defaults to sqrt of --typical-acceptance-sampler-posterior-threshold i.e. 0.3
  --disable-logprobs-during-spec-decoding DISABLE_LOGPROBS_DURING_SPEC_DECODING
                        Category: Speculative Decoding Options If set to True, token log probabilities are not returned during speculative
                        decoding. If set to False, log probabilities are returned according to the settings in SamplingParams. If not
                        specified, it defaults to True. Disabling log probabilities during speculative decoding reduces latency by skipping
                        logprob calculation in proposal sampling, target sampling, and after accepted tokens are determined.
  --enable-lora         Category: Adapter Options If True, enable handling of LoRA adapters.
  --enable-lora-bias    If True, enable bias for LoRA adapters.
  --max-loras MAX_LORAS
                        Category: Adapter Options Max number of LoRAs in a single batch.
  --max-lora-rank MAX_LORA_RANK
                        Category: Adapter Options Max LoRA rank.
  --lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE
                        Category: Adapter Options Maximum size of extra vocabulary that can be present in a LoRA adapter (added to the base
                        model vocabulary).
  --lora-dtype {auto,float16,bfloat16,float32}
                        Category: Adapter Options Data type for LoRA. If auto, will default to base model dtype.
  --max-cpu-loras MAX_CPU_LORAS
                        Category: Adapter Options Maximum number of LoRAs to store in CPU memory. Must be >= than max_num_seqs. Defaults to
                        max_num_seqs.
  --long-lora-scaling-factors LONG_LORA_SCALING_FACTORS
                        Category: Adapter Options Specify multiple scaling factors (which can be different from base model scaling factor -
                        see eg. Long LoRA) to allow for multiple LoRA adapters trained with those scaling factors to be used at the same time.
                        If not specified, only adapters trained with the base model scaling factor are allowed.
  --fully-sharded-loras
                        Category: Adapter Options By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling
                        this will use the fully sharded layers. At high sequence length, max rank or tensor parallel size, this is likely
                        faster.
  --qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH
                        Category: Adapter Options Name or path of the LoRA adapter to use.
  --enable-prompt-adapter
                        Category: Adapter Options If True, enable handling of PromptAdapters.
  --max-prompt-adapters MAX_PROMPT_ADAPTERS
                        Category: Adapter Options Max number of PromptAdapters in a batch.
  --max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN
                        Category: Adapter Options Max number of PromptAdapters tokens
  --disable-log-stats   Category: Log Options disable logging statistics
  --disable-async-output-proc
                        Disable async output processing. THis may result in lower performance.
  --override-neuron-config OVERRIDE_NEURON_CONFIG
                        Override or set neuron device configuration. e.g. {"cast_logits_dtype": "bloat16"}.'
  --scheduling-policy {fcfs,priority}
                        The scheduling policy to use. "fcfs" (first come first served, i.e. requests are handled in order of arrival; default)
                        or "priority" (requests are handled based on given priority (lower value means earlier handling) and time of arrival
                        deciding any ties).
  --disable-log-requests
                        Disable logging requests.
  --uvloop              Use the Uvloop asyncio event loop to possibly increase performance
```

## Tool Calling in the chat completions API

### Named Function Calling

Aphrodite supports only named function calling in the chat completion API by default. It does so using Outlines, so this is enabled by default, and will work with any supported model. You are guaranteed a validly-parsable function call - not a high-quality one.

To use a named function, you need to define the functions in the `tools` parameter of the chat completions request, and specify the `name` of one of the tools in the `tool_choice` parameter of the chat completion request.

### Automatic Function Calling

To enable this feature, you should set the following flags:

- `--enable-auto-tool-choice` (Required): This tells Aphrodite that you want to allow the model to generate its own tool calls when it deems appropriate.
- `--tool-call-parser`: The tool parser to use. Currently, only `hermes`, `mistral`, `llama3_json`, and `internlm` are supported. More will be added soon, but you may also register your own. Please see [Tool Plugin](/developer/tool-plugin) for more info.
- `--tool-parser-plugin`: The tool parser plugin to use. See above.
- `--chat-template`: The chat template to handle `tool`-role messages and `assistant`-role messages that contain previously generated tool calls. We provide these for `hermes`, `mistral`, `llama3_json`, and `internlm` in the examples/chat_templates directory.

### Hermes Models

All NousResearch Hermes-series models newer than Hermes 2 Pro are supported:

- `NousResearch/Hermes-2-Pro-*`
- `NousResearch/Hermes-2-Theta-*`
- `NousResearch/Hermes-3-*`

:::tip
Hermes 2 Theta models are known to have degraded tool call quality due to the merge step in their creation.
:::

Enable with `--tool-call-parser hermes --chat-template examples/chat_templates/hermes_tool.jinja`

### Mistral Models

Supported models:

- `mistralai/Mistral-7B-Instruct-v0.3`
- Additional mistral function-calling models are supported as well.

Known issues:

1. Mistral 7B struggles to generate parallel tool calls correctly.
2. Mistral's `tokenizer_config.json` chat template requires tool call IDs that are exactly 9 digits, which is much shorter than what Aphrodite generates. Since an exception is thrown when this condition is not met, we provide custom chat templtes:

- `examples/chat_templates/mistral_tool.jinja`: this is the "official" Mistral chat template, but tweaked so that is works with Aphrodite's tool call IDs (provided `tool_call_id` fields are truncated to the last 9 digits).
- `examples/chat_templates/mistral_parallel_tool.jinja`: this is a "better" version that adds a tool-use system prompt when tools are provided, that results in much better reliability when working with parallel tool calling.

Enable with `--tool-call-parser mistral --chat-template examples/chat_templates/mistral_tool.jinja`

### Llama Models

Supported models:

- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `meta-llama/Meta-Llama-3.1-405B-Instruct`
- `meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`

The tool calling supported is the [JSON based tool calling](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling). Other tool calling formats like the built-in python tool calling or custom tool calling are not supported.

Known issues:

1. Parallel tool calls are not supported.
2. The model can generate parameters with a wrong format, such as generating an array serialized as string instead of an array.

We provide two chat templates: `examples/chat_templates/llama_3.1_json_tool.jinja` and `examples/chat_templates/llama_3.2_json_tool.jinja`, for both versions of the Llama-3 models that support function calling.

Enable with: `--tool-call-parser llama3_json --chat-template examples/chat_templates/llama_3.2_json_tool.jinja`

### InternLM Models

Supported models:

- `internlm/internlm2_5-7b-chat`
- Additional models that support function calling are supported as well.

Known issues:

- This implementation technically supports InternLM2 as well, but they're not that stable.

Enable with: `--tool-call-parser internlm --chat-template examples/chat_templates/internlm2_tools.jinja`

#### Jamba Models

AI21's Jamba-1.5 models are supported.

- `ai21labs/AI21-Jamba-1.6-Mini`
- `ai21labs/AI21-Jamba-1.6-Large`

Flags: `--tool-call-parser jamba`
