# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import logging
import os
import sys
import tempfile
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    SONAR_HOST_IP: str = ""
    SONAR_PORT: int | None = None
    SONAR_RPC_BASE_PATH: str = tempfile.gettempdir()
    SONAR_USE_MODELSCOPE: bool = False
    SONAR_USE_FASTOKENS: bool = False
    SONAR_RINGBUFFER_WARNING_INTERVAL: int = 60
    SONAR_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    SONAR_ROCM_SLEEP_MEM_CHUNK_SIZE: int = 256
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    SONAR_ENGINE_READY_TIMEOUT_S: int = 600
    SONAR_CHAT_TEMPLATE_RENDER_TIMEOUT: float = 30.0
    SONAR_CHAT_TEMPLATE_FUEL: int = 500000
    SONAR_API_KEY: str | None = None
    SONAR_DEBUG_LOG_API_SERVER_RESPONSE: bool = False
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_ENDPOINT_URL: str | None = None
    SONAR_MODEL_REDIRECT_PATH: str | None = None
    SONAR_CACHE_ROOT: str = os.path.expanduser("~/.cache/sonar")
    SONAR_CONFIG_ROOT: str = os.path.expanduser("~/.config/sonar")
    SONAR_USAGE_STATS_SERVER: str = ""
    SONAR_NO_USAGE_STATS: bool = False
    SONAR_DO_NOT_TRACK: bool = False
    SONAR_USAGE_SOURCE: str = "production"
    SONAR_CONFIGURE_LOGGING: bool = True
    SONAR_LOGGING_LEVEL: str = "INFO"
    SONAR_LOGGING_PREFIX: str = ""
    SONAR_LOGGING_STREAM: str = "ext://sys.stdout"
    SONAR_LOGGING_CONFIG_PATH: str | None = None
    SONAR_LOGGING_COLOR: str = "auto"
    SONAR_LOGGING_VERBOSE: bool = False
    NO_COLOR: bool = False
    FORCE_COLOR: bool = False
    SONAR_LOG_STATS_INTERVAL: float = 10.0
    SONAR_REQUEST_LEVEL_METRICS: bool = True
    SONAR_TRACE_FUNCTION: int = 0
    SONAR_USE_FLASHINFER_SAMPLER: bool = True
    SONAR_PP_LAYER_PARTITION: str | None = None
    SONAR_CPU_KVCACHE_SPACE: int | None = None
    SONAR_CPU_OMP_THREADS_BIND: str = "auto"
    SONAR_CPU_NUM_OF_RESERVED_CPU: int | None = None
    SONAR_CPU_CI_ENV: bool = False
    SONAR_CPU_ATTN_SPLIT_KV: bool = True
    SONAR_ZENTORCH_WEIGHT_PREPACK: bool = True
    SONAR_CPU_INT4_W4A8: bool = True
    SONAR_XLA_CACHE_PATH: str = os.path.join(SONAR_CACHE_ROOT, "xla_cache")
    SONAR_XLA_CHECK_RECOMPILATION: bool = False
    SONAR_USE_FUSED_MQA_TOPK: bool = True
    SONAR_SPARSE_INDEXER_MAX_LOGITS_MB: int = 512
    SONAR_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB: int = 2048
    SONAR_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN: int = 8192
    SONAR_XPU_PP_MICROBATCH: bool = True
    SONAR_DISTRIBUTED_USE_SPLIT_GROUP: bool = False
    SONAR_XLA_USE_SPMD: bool = False
    SONAR_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"
    SONAR_ASSETS_CACHE: str = os.path.join(SONAR_CACHE_ROOT, "assets")
    SONAR_ASSETS_CACHE_MODEL_CLEAN: bool = False
    SONAR_IMAGE_FETCH_TIMEOUT: int = 5
    SONAR_VIDEO_FETCH_TIMEOUT: int = 30
    SONAR_AUDIO_FETCH_TIMEOUT: int = 10
    SONAR_MEDIA_CACHE: str = ""
    SONAR_MEDIA_CACHE_MAX_SIZE_MB: int = 5120
    SONAR_MEDIA_CACHE_TTL_HOURS: float = 24
    SONAR_MEDIA_FETCH_MAX_RETRIES: int = 3
    SONAR_MAX_MEDIA_DOWNLOAD_SIZE_MB: int = 256
    SONAR_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB: int = 256
    SONAR_MEDIA_URL_ALLOW_REDIRECTS: bool = True
    SONAR_MEDIA_LOADING_THREAD_COUNT: int = 8
    SONAR_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
    SONAR_MAX_AUDIO_DECODE_DURATION_S: int = 600
    SONAR_MAX_AUDIO_DECODE_BYTES: int = 268_435_456
    SONAR_MAX_AUDIO_PREPROCESS_WORKERS: int = max(1, min(os.cpu_count() or 1, 2))
    SONAR_MAX_EMBED_DECODE_BYTES: int = 2_147_483_648
    SONAR_MAX_IMAGE_PIXELS: int = 178_956_970
    SONAR_MAX_MEDIA_REQUEST_DECODE_BYTES: int = 715_827_880
    SONAR_VIDEO_LOADER_BACKEND: str = "opencv"
    SONAR_MEDIA_CONNECTOR: str = "http"
    SONAR_TARGET_DEVICE: str = "cuda"
    SONAR_USE_PRECOMPILED: bool = False
    SONAR_PRECOMPILED_WHEEL_LOCATION: str | None = None
    SONAR_PRECOMPILED_WHEEL_COMMIT: str | None = None
    SONAR_SKIP_PRECOMPILED_VERSION_SUFFIX: bool = False
    SONAR_MAIN_CUDA_VERSION: str = "13.2"
    SONAR_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"] = "highest"
    SONAR_BATCH_INVARIANT: bool = False
    SONAR_TRITON_USE_TD: bool | None = None
    SONAR_GPU_SYNC_CHECK: Literal["warn", "error"] | None = None
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    SONAR_DOCKER_BUILD_CONTEXT: bool = False
    SONAR_BUILD_COMMIT: str = "unknown"
    SONAR_BUILD_PIPELINE: str = "local"
    SONAR_BUILD_URL: str = ""
    SONAR_IMAGE_TAG: str = ""
    SONAR_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None = None
    VERBOSE: bool = False
    SONAR_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    SONAR_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds
    SONAR_HTTP_MAX_REQUEST_BODY_SIZE: int = 32 * 1024 * 1024  # bytes
    SONAR_MAX_N_SEQUENCES: int = 16384
    SONAR_MAX_COMPLETION_PROMPTS: int = 1024
    SONAR_MAX_STOP_STRINGS: int = 4
    SONAR_MAX_NUM_BAD_WORDS: int = 128
    SONAR_MAX_BAD_WORDS_TOTAL_TOKENS: int = 1024
    SONAR_PLUGINS: list[str] | None = None
    SONAR_LORA_RESOLVER_CACHE_DIR: str | None = None
    SONAR_LORA_RESOLVER_HF_REPO_LIST: str | None = None
    SONAR_USE_AOT_COMPILE: bool = False
    SONAR_USE_BYTECODE_HOOK: bool = True
    SONAR_FORCE_AOT_LOAD: bool = False
    SONAR_USE_MEGA_AOT_ARTIFACT: bool = False
    SONAR_COMPILE_ON_ONE_RANK: bool = False
    SONAR_USE_TRITON_AWQ: bool = False
    SONAR_FASTSAFETENSORS_QUEUE_SIZE: int = 0
    SONAR_TRITON_FORCE_FIRST_CONFIG: bool = False
    SONAR_TRITON_JIT_WARMUP_NUM_THREADS: int = 4
    SONAR_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    SONAR_SKIP_P2P_CHECK: bool = True
    SONAR_PP_TRANSPORT: Literal["auto", "nccl", "copy_engine"] = "auto"
    SONAR_DISABLED_KERNELS: list[str] = []
    SONAR_USE_HW_AGNOSTIC: bool = False
    SONAR_ENABLE_FLA_PACKED_RECURRENT_DECODE: bool = True
    SONAR_GDN_DECODE_KERNEL: Literal["cuda", "triton"] = "cuda"
    SONAR_DISABLE_PYNCCL: bool = False
    SONAR_USE_OINK_OPS: bool = False
    SONAR_MXFP4_EMULATION_DEQUANT_AT_LOAD: bool = False
    SONAR_MXFP4_COMPRESS_SCALES: bool = False
    SONAR_MXFP8_EMULATION_DEQUANT_AT_LOAD: bool = True
    SONAR_ROCM_USE_AITER: bool = False
    SONAR_ROCM_USE_AITER_CUSTOM_AR: bool = True
    SONAR_ROCM_MUSE_MOE_REDUCE_DTYPE: str = "fp32"
    SONAR_ROCM_MUSE_MOE_ONLINE_QUANT: str = ""
    SONAR_ROCM_USE_AITER_LINEAR: bool = True
    SONAR_ROCM_USE_AITER_LINEAR_HIPBMM: bool = False
    SONAR_ROCM_USE_AITER_MOE: bool = True
    SONAR_ROCM_AITER_MOE_DISPATCH_POLICY: int = 0
    SONAR_ROCM_USE_AITER_MOE_A4W4_DSV4: bool | None = None
    SONAR_ROCM_USE_AITER_MOE_SITUV2: Literal["auto", "a4w4", "a8w4", "a16w4"] = "auto"
    SONAR_ROCM_USE_AITER_RMSNORM: bool = True
    SONAR_ROCM_USE_AITER_MLA: bool = True
    SONAR_ROCM_AITER_MLA_ASM_PADDING: Literal["auto", "gluon", "asm"] = "auto"
    SONAR_ROCM_USE_AITER_MHA: bool = True
    SONAR_ROCM_USE_AITER_TRITON_SPARSE_MLA: bool = False
    SONAR_ROCM_USE_AITER_FLYDSL_SPARSE_MLA: bool = False
    SONAR_ROCM_AITER_FLYDSL_SPARSE_MLA_MIN_TOKENS: int = 160
    SONAR_ROCM_USE_AITER_TRITON_ROPE: bool = False
    SONAR_ROCM_USE_AITER_FP8BMM: bool = False
    SONAR_ROCM_USE_AITER_FP4BMM: bool = False
    SONAR_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False
    SONAR_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = False
    SONAR_ROCM_USE_AITER_TRITON_GEMM: bool = True
    SONAR_ROCM_USE_SKINNY_GEMM: bool = True
    SONAR_ROCM_FP8_PADDING: bool = True
    SONAR_ROCM_MOE_PADDING: bool = True
    SONAR_ROCM_SHUFFLE_KV_CACHE_LAYOUT: bool = False
    SONAR_ROCM_PARALLEL_INDEX_TOPK: bool = False
    SONAR_ENABLE_V1_MULTIPROCESSING: bool = True
    SONAR_LOG_BATCHSIZE_INTERVAL: float = -1
    SONAR_DISABLE_COMPILE_CACHE: bool = False
    SONAR_REPLICATE_EMBED: bool = False
    SONAR_USE_LAYERNAME: bool = True
    SONAR_USE_RUST_FRONTEND: bool = False
    SONAR_USE_RUST_BENCH: bool = False
    SONAR_RUST_FRONTEND_PATH: str | None = "auto"
    SONAR_SERVER_DEV_MODE: bool = False
    SONAR_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    SONAR_MLA_DISABLE: bool = False
    SONAR_RAY_PER_WORKER_GPUS: float = 1.0
    SONAR_RAY_BUNDLE_INDICES: str = ""
    SONAR_CUDART_SO_PATH: str | None = None
    SONAR_DP_RANK: int = 0
    SONAR_DP_RANK_LOCAL: int = -1
    SONAR_DP_SIZE: int = 1
    SONAR_USE_STANDALONE_COMPILE: bool = True
    SONAR_ENABLE_PREGRAD_PASSES: bool = True
    SONAR_USE_BREAKABLE_CUDAGRAPH: bool = False
    SONAR_DP_MASTER_IP: str = "127.0.0.1"
    SONAR_DP_MASTER_PORT: int = 0
    SONAR_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    SONAR_RAY_DP_PACK_STRATEGY: Literal["strict", "fill", "span"] = "strict"
    SONAR_RAY_DP_PLACEMENT_NODE_IPS: str = ""
    SONAR_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY: str = ""
    SONAR_RAY_EXTRA_ENV_VARS_TO_COPY: str = ""
    SONAR_MARLIN_USE_ATOMIC_ADD: bool = False
    SONAR_HUMMING_ONLINE_QUANT_CONFIG: dict[str, Any] | None = None
    SONAR_HUMMING_INPUT_QUANT_CONFIG: dict[str, Any] | None = None
    SONAR_HUMMING_USE_F16_ACCUM: bool = False
    SONAR_HUMMING_MOE_GEMM_TYPE: Literal["indexed", "grouped", "auto"] | None = None
    SONAR_B12X_MOE_FP4_FORCE_A16: bool = False
    SONAR_DSV41_SM120_NATIVE: bool = False
    SONAR_DEEPEPLL_NVFP4_DISPATCH: bool = False
    SONAR_V1_USE_OUTLINES_CACHE: bool = False
    SONAR_TPU_USING_PATHWAYS: bool = False
    SONAR_USE_DEEP_GEMM: bool = True
    SONAR_MOE_USE_DEEP_GEMM: bool = True
    SONAR_USE_DEEP_GEMM_E8M0: bool = True
    SONAR_USE_DEEP_GEMM_TMA_ALIGNED_SCALES: bool = True
    SONAR_DCP_Q_REPLICATE: bool = False
    SONAR_USE_DIRECT_DCP_A2A: bool | None = None
    SONAR_USE_DIRECT_DCP_Q_GATHER: bool | None = None
    SONAR_USE_DIRECT_DCP_KV_GATHER: bool | None = None
    SONAR_DEEP_GEMM_WARMUP: Literal[
        "skip",
        "full",
        "relax",
    ] = "relax"
    SONAR_USE_FUSED_MOE_GROUPED_TOPK: bool = True
    SONAR_MOE_SKIP_PADDING: bool = True
    SONAR_STEP5_SHARD_SP_SHARED_EXPERT: bool = False
    SONAR_KIMI_K3_SHARD_SP_SHARED_EXPERT: bool = False
    SONAR_KIMI_K3_AUX_ATTN_RES_STREAM: bool = False
    SONAR_KIMI_K3_GEMM_AR: bool = True
    SONAR_ENABLE_GEMM_RS: bool = False
    SONAR_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool = True
    SONAR_USE_FLASHINFER_MOE_INT4: bool = False
    SONAR_USE_LOCALITY_DOMAINS: bool = False
    SONAR_MOE_LOCALIZE_BACKFILL: bool = False
    SONAR_FLASHINFER_CUTEDSL_MOE_LOCALIZE_MEMSET_OVERLAP: bool = True
    SONAR_FLASHINFER_CUTEDSL_MOE_LOCALIZE_FREE_FULL_WEIGHTS: bool = False
    SONAR_FLASHINFER_AUTOTUNE_CACHE_DIR: str | None = None
    SONAR_FLASHINFER_AUTOTUNE_SKIP_OPS: list[str] | None = None
    SONAR_FLASHINFER_ALLREDUCE_BACKEND: Literal["auto", "trtllm", "mnnvl"] = "auto"
    SONAR_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024
    SONAR_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE: bool = False
    SONAR_XGRAMMAR_CACHE_MB: int = 512
    SONAR_REGEX_COMPILATION_TIMEOUT_S: int = 5
    SONAR_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    SONAR_ALLOW_INSECURE_SERIALIZATION: bool = False
    SONAR_DISABLE_REQUEST_ID_RANDOMIZATION: bool = False
    SONAR_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    SONAR_NIXL_SIDE_CHANNEL_PORT: int = 5600
    SONAR_P2P_SIDE_CHANNEL_HOST: str = "localhost"
    SONAR_P2P_SIDE_CHANNEL_PORT: int = 5710
    SONAR_P2P_MAX_PEERS: int = 128
    SONAR_P2P_HANDSHAKE_TIMEOUT_S: int = 30
    SONAR_P2P_IDLE_TIMEOUT_S: int = 300
    SONAR_P2P_TRANSFER_STALL_TIMEOUT_S: float = 20.0
    SONAR_EC_SIDE_CHANNEL_HOST: str = ""
    SONAR_EC_SIDE_CHANNEL_PORT: int = 0
    SONAR_MOONCAKE_BOOTSTRAP_PORT: int = 8998
    SONAR_MOONCAKE_STORE_TIER_LOG: bool = False
    SONAR_MOONCAKE_STORE_SELECTIVE_LEASE: bool = False
    SONAR_MOONCAKE_LOAD_RECV_THREADS: int = 1
    SONAR_MOONCAKE_DISK_STAGING_USABLE_RATIO: float = 0.9
    MOONCAKE_PREFERRED_SEGMENT: str | None = None
    MOONCAKE_REQUESTER_LOCAL_HOSTNAME: str | None = None
    SONAR_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
    SONAR_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    SONAR_ENFORCE_STRICT_TOOL_CALLING: bool = True
    SONAR_BEAM_ALWAYS_CONSTRAIN_OUTPUT: bool = True
    SONAR_MQ_MAX_CHUNK_BYTES_MB: int = 16
    SONAR_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    SONAR_WORKER_SHUTDOWN_TIMEOUT_SECONDS: int = 5
    SONAR_KV_CACHE_LAYOUT: Literal["LBNHC", "LBHNC", "LHBNC", "NHD", "HND", "BLHNC", "BLNHC", "BHLNC"] | None = None
    SONAR_SSM_CONV_STATE_LAYOUT: Literal["SD", "DS"] | None = None
    SONAR_COMPUTE_NANS_IN_LOGITS: bool = False
    SONAR_RAISE_ON_LOGIT_NANS: bool = False
    SONAR_CONFIDENTIAL_COMPUTE: str | None = None
    SONAR_ROCM_QUICK_REDUCE_QUANTIZATION: Literal["FP", "INT8", "INT6", "INT4", "INT3", "NONE"] = "NONE"
    SONAR_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    SONAR_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None = None
    SONAR_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB: int | None = None
    SONAR_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB: int | None = None
    SONAR_MOONCAKE_ABORT_REQUEST_TIMEOUT: int = 480
    SONAR_MOONCAKE_CONNECTOR_TIMEOUT: float = 30.0
    SONAR_ENABLE_CUDAGRAPH_GC: bool = False
    SONAR_LOOPBACK_IP: str = ""
    SONAR_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = True
    SONAR_ENABLE_RESPONSES_API_STORE: bool = False
    SONAR_ENABLE_COHERE_API: bool = False
    SONAR_HAS_FLASHINFER_CUBIN: bool = False
    SONAR_ROCM_FP8_MFMA_PAGE_ATTN: bool = False
    SONAR_ALLREDUCE_USE_SYMM_MEM: bool = True
    SONAR_ALLREDUCE_USE_FLASHINFER: bool = True
    SONAR_ALLREDUCE_USE_FLASHINFER_PCIE_IPC: bool = False
    SONAR_ALLREDUCE_USE_PCIE_PULL: bool = False
    SONAR_ALLREDUCE_PCIE_TRANSPORT: Literal["paired", "host"] = "paired"
    SONAR_TUNED_CONFIG_FOLDER: str | None = None
    SONAR_ENABLE_STARTUP_PLAN: bool = False
    SONAR_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: set[str] = set()
    SONAR_USE_EXPERIMENTAL_PARSER_CONTEXT: bool = False
    SONAR_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: bool = False
    SONAR_SYSTEM_START_DATE: str | None = None
    SONAR_TOOL_JSON_ERROR_AUTOMATIC_RETRY: bool = False
    SONAR_CUSTOM_SCOPES_FOR_PROFILING: bool = False
    SONAR_NVTX_SCOPES_FOR_PROFILING: bool = False
    SONAR_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = True
    SONAR_OBJECT_STORAGE_SHM_BUFFER_NAME: str = "SONAR_OBJECT_STORAGE_SHM_BUFFER"
    SONAR_DEEPEP_BUFFER_SIZE_MB: int = 1024
    SONAR_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE: bool = False
    SONAR_MORI_EP_V2: bool | None = None
    SONAR_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD: int = 256
    SONAR_DEEPEP_LOW_LATENCY_USE_MNNVL: bool = False
    SONAR_DEEPEP_V2_ALLOW_HYBRID_MODE: bool = True
    SONAR_DEEPEP_V2_PREFER_OVERLAP: bool = False
    SONAR_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION: bool = False
    SONAR_DBO_COMM_SMS: int = 20
    SONAR_PATTERN_MATCH_DEBUG: str | None = None
    SONAR_DEBUG_DUMP_PATH: str | None = None
    SONAR_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool = True
    SONAR_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool = True
    SONAR_USE_NCCL_SYMM_MEM: bool = False
    SONAR_NCCL_INCLUDE_PATH: str | None = None
    SONAR_GC_DEBUG: str = ""
    SONAR_DEBUG_WORKSPACE: bool = False
    SONAR_DISABLE_SHARED_EXPERTS_STREAM: bool = False
    SONAR_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION: bool = False
    SONAR_SKINNY_GEMM_MAX_WEIGHT_BYTES: int = 0
    SONAR_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: int = 256
    SONAR_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: int = 1024
    SONAR_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD: int = 256
    SONAR_COMPILE_CACHE_SAVE_FORMAT: Literal["binary", "unpacked"] = "binary"
    SONAR_LOG_MODEL_INSPECTION: bool = False
    SONAR_DEBUG_MFU_METRICS: bool = False
    SONAR_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY: bool = False
    SONAR_WEIGHT_OFFLOADING_DISABLE_UVA: bool = False
    SONAR_RELOAD_MODE: Literal["layerwise", "direct"] = "layerwise"
    SONAR_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS: int = 0
    SONAR_WSL2_ENABLE_PIN_MEMORY: bool = False
    SONAR_DISABLE_LOG_LOGO: bool = False
    SONAR_DISABLE_SM89_DSA: bool = False
    SONAR_LORA_DISABLE_PDL: bool = False
    SONAR_ENABLE_CUDA_COMPATIBILITY: bool = False
    SONAR_CUDA_COMPATIBILITY_PATH: str | None = None
    SONAR_SKIP_MODEL_NAME_VALIDATION: bool = False
    """If set, Sonar will skip model name validation in API requests.
    This allows any model name to be accepted in the 'model' field of requests,
    making the server model-name agnostic. Useful for proxy/gateway scenarios."""
    SONAR_LOG_ROUTES: Literal["off", "compact", "full"] = "compact"
    SONAR_ELASTIC_EP_SCALE_UP_LAUNCH: bool = False
    SONAR_ELASTIC_EP_DRAIN_REQUESTS: bool = False
    SONAR_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool = True
    SONAR_XPU_FORCE_N_CONTIG_WEIGHT: bool = False
    SONAR_XPU_USE_SAMPLER_KERNEL: bool = True
    SONAR_XPU_INC_WNA16_BACKEND: Literal["auto", "ark", "w4a16", "w4a8"] = "auto"
    SONAR_LORA_ENABLE_DUAL_STREAM: bool = False
    SONAR_GPU_NIC_PCIE_MAPPING: str = ""
    SONAR_NIC_SELECTION_VARS: str = ""
    SONAR_ENABLE_HPC_OPS: bool = False
    SONAR_REQUEST_BODY_LOG_LEVEL: str = "DEBUG"
    SONAR_TEST_FORCE_LOAD_FORMAT: str = "dummy"
    SONAR_MOE_ROUTING_SIMULATION_STRATEGY: str = ""
    SONAR_PROCESS_NAME_PREFIX: str = "SONAR"
    SONAR_USE_SIMPLE_KV_OFFLOAD: bool = False
    SONAR_USE_SPINLOOP_EXT: bool = False
    SONAR_SKIP_VERSION_SUFFIX: bool = False


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_json_str_or_file(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if os.path.exists(value):
        with open(value) as f:
            return json.load(f)
    return json.loads(value)


def disable_compile_cache() -> bool:
    return bool(int(os.getenv("SONAR_DISABLE_COMPILE_CACHE", "0")))


def use_aot_compile() -> bool:
    from sonar.utils.torch_version import is_torch_equal_or_newer

    default_value = "1" if is_torch_equal_or_newer("2.10.0") and not disable_compile_cache() else "0"

    return os.environ.get("SONAR_USE_AOT_COMPILE", default_value) == "1"


def use_mega_aot_artifact():
    from sonar.utils.torch_version import is_torch_equal_or_newer

    default_value = "1" if is_torch_equal_or_newer("2.12.0.dev") and use_aot_compile() else "0"

    return os.environ.get("SONAR_USE_MEGA_AOT_ARTIFACT", default_value) == "1"


def compile_on_one_rank() -> bool:
    from sonar.platforms import current_platform
    from sonar.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.14.0.dev"):
        return False
    # Inductor implements the mode for CUDA/ROCm and XPU only.
    if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
        return False
    return os.environ.get("SONAR_COMPILE_ON_ONE_RANK", "0") == "1"


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """Create a lambda that validates environment variable against allowed choices.

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict

    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(f"Invalid value '{value}' for {env_name}. Valid options: {actual_choices}.")

        return value

    return _get_validated_env


def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]:
    """Create a lambda that validates environment variable
    containing comma-separated values against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default list of values if not set
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables
        dict that returns list of strings

    """

    def _get_validated_env_list() -> list[str]:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Split comma-separated values and strip whitespace
        values = [v.strip() for v in value.split(",") if v.strip()]

        if not values:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        # Validate each value
        for val in values:
            if not case_sensitive:
                check_value = val.lower()
                check_choices = [choice.lower() for choice in actual_choices]
            else:
                check_value = val
                check_choices = actual_choices

            if check_value not in check_choices:
                raise ValueError(f"Invalid value '{val}' in {env_name}. Valid options: {actual_choices}.")

        return values

    return _get_validated_env_list


def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]:
    """Creates a lambda which that validates environment variable
    containing comma-separated values against allowed choices which
    returns choices as a set.
    """

    def _get_validated_env_set() -> set[str]:
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())

    return _get_validated_env_set


def get_sonar_port() -> int | None:
    """Get the port from SONAR_PORT environment variable.

    Returns:
        The port number as an integer if SONAR_PORT is set, None otherwise.

    Raises:
        ValueError: If SONAR_PORT is a URI, suggest k8s service discovery issue.

    """
    if "SONAR_PORT" not in os.environ:
        return None

    port = os.getenv("SONAR_PORT", "0")

    try:
        return int(port)
    except ValueError as err:
        from urllib3.util import parse_url

        parsed = parse_url(port)
        if parsed.scheme:
            raise ValueError(
                f"SONAR_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://sonar.dphn.ai/en/latest/configuration/env_vars.html"
            ) from None
        raise ValueError(f"SONAR_PORT '{port}' must be a valid integer") from err


def _generate_shm_name() -> str:
    import pybase64

    # Fit macOS's 30-character limit without truncating the UUID.
    encoded_uuid = pybase64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii")
    return "sonar_" + encoded_uuid.rstrip("=")


def get_env_or_set_default(
    env_name: str,
    default_factory: Callable[[], str],
) -> Callable[[], str]:
    """Create a lambda that returns an environment variable value if set,
    or generates and sets a default value using the provided factory function.
    """

    def _get_or_set_default() -> str:
        value = os.getenv(env_name)
        if value is not None:
            return value

        default_value = default_factory()
        os.environ[env_name] = default_value
        return default_value

    return _get_or_set_default


# The start-* and end* here are used by the documentation generator
# to extract the used env vars.

# --8<-- [start:env-vars-definition]

logger = logging.getLogger(__name__)


def _resolve_rust_cli_path() -> str | None:
    """Resolve the sonar-rs binary path.

    Returns None unless SONAR_USE_RUST_FRONTEND or
    SONAR_USE_RUST_BENCH is enabled.
    When enabled, resolves SONAR_RUST_FRONTEND_PATH ("auto" by default)
    to the actual binary path.
    """
    use_rust = bool(int(os.environ.get("SONAR_USE_RUST_FRONTEND", "0"))) or bool(
        int(os.environ.get("SONAR_USE_RUST_BENCH", "0"))
    )
    raw = os.environ.get("SONAR_RUST_FRONTEND_PATH", "auto")

    if not use_rust:
        if os.environ.get("SONAR_RUST_FRONTEND_PATH") is not None:
            logger.warning(
                "SONAR_RUST_FRONTEND_PATH is set without enabling "
                "SONAR_USE_RUST_FRONTEND or SONAR_USE_RUST_BENCH. "
                "Set one of them to 1 to use the sonar-rs binary."
            )
        return None

    if raw.lower() in ("auto", "1", "true"):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(pkg_dir, "sonar-rs")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

        raise FileNotFoundError(
            "SONAR_RUST_FRONTEND_PATH=auto but the sonar-rs binary was "
            f"not found at {candidate}. "
            "Build with setuptools-rust or set the path explicitly."
        )
    return raw


# Environment variables that were removed, mapped to a migration hint. Setting
# one is an error rather than an "unknown variable" warning.
removed_environment_variables: dict[str, str] = {
    "SONAR_MARLIN_INPUT_DTYPE": (
        "Marlin no longer supports 8-bit activations. Use "
        "`--quantization humming` and set the activation dtype with "
        'SONAR_HUMMING_INPUT_QUANT_CONFIG, e.g. \'{"dtype":"int8"}\'.'
    ),
}

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of Sonar, supporting [cuda (by default),
    # rocm, cpu]
    "SONAR_TARGET_DEVICE": lambda: os.getenv("SONAR_TARGET_DEVICE", "cuda").lower(),
    # Build-time only: skip all native builds (CMake/CUDA and Rust); binaries must
    # be downloaded from the matching main-branch wheel.
    "SONAR_USE_PRECOMPILED": lambda: os.getenv("SONAR_USE_PRECOMPILED", "").lower() in ("1", "true", "yes"),
    # Exact local path or URL to use instead of resolving a per-commit wheel.
    "SONAR_PRECOMPILED_WHEEL_LOCATION": lambda: os.getenv("SONAR_PRECOMPILED_WHEEL_LOCATION"),
    # Full main-branch commit SHA, or "nightly", used instead of the checkout's
    # merge-base with main.
    "SONAR_PRECOMPILED_WHEEL_COMMIT": lambda: os.getenv("SONAR_PRECOMPILED_WHEEL_COMMIT"),
    # Preserve the source-derived version in packaging contexts that consume
    # precompiled artifacts internally.
    "SONAR_SKIP_PRECOMPILED_VERSION_SUFFIX": lambda: (
        os.getenv("SONAR_SKIP_PRECOMPILED_VERSION_SUFFIX", "").lower() in ("1", "true", "yes")
    ),
    # Main CUDA version of Sonar. This follows PyTorch but can be overridden.
    "SONAR_MAIN_CUDA_VERSION": lambda: os.getenv("SONAR_MAIN_CUDA_VERSION", "").lower() or "13.2",
    # Controls PyTorch float32 matmul precision mode within Sonar workers.
    # Valid options mirror torch.set_float32_matmul_precision
    "SONAR_FLOAT32_MATMUL_PRECISION": env_with_choices(
        "SONAR_FLOAT32_MATMUL_PRECISION",
        "highest",
        ["highest", "high", "medium"],
        case_sensitive=False,
    ),
    # Enable batch-invariant mode: deterministic results regardless of
    # batch composition. Requires NVIDIA GPU with compute capability >= 9.0.
    "SONAR_BATCH_INVARIANT": lambda: bool(int(os.getenv("SONAR_BATCH_INVARIANT", "0"))),
    "SONAR_REPLICATE_EMBED": lambda: os.getenv("SONAR_REPLICATE_EMBED", "0").strip().lower() in ("1", "true"),
    # Use tensor descriptors for Q/K/V loads and output stores in the
    # Triton unified-attention kernel.  Enables HW 2D block reads on
    # Intel XPU; the non-TD branch is dead-code-eliminated at Triton
    # compile time so other platforms see no overhead.  Tri-state override:
    # unset (default) lets the `triton_attn` backend auto-select per
    # platform (currently auto-enabled on XPU only); ``1`` forces TD on;
    # ``0`` forces TD off.  Useful for A/B benchmarking the TD path.
    "SONAR_TRITON_USE_TD": lambda: {"1": True, "0": False}.get(os.getenv("SONAR_TRITON_USE_TD", "").strip()),
    # If set, enable GPU<->CPU synchronization checking around the worker's
    # `execute_model` and `sample_tokens` calls, via PyTorch's sync debug mode
    # plus wrappers flagging `non_blocking` CPU<->CUDA copies that silently
    # block the host (CPU tensors that are not pinned or not densely laid out).
    # Valid values are "warn" (warn on each sync) or "error" (raise on sync).
    # Unset disables the check. See `torch.cuda.set_sync_debug_mode`.
    "SONAR_GPU_SYNC_CHECK": env_with_choices("SONAR_GPU_SYNC_CHECK", None, ["warn", "error"]),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # Used to mark that setup.py is running in a Docker build context,
    # in order to force the use of precompiled binaries.
    "SONAR_DOCKER_BUILD_CONTEXT": lambda: (
        os.environ.get("SONAR_DOCKER_BUILD_CONTEXT", "").strip().lower() in ("1", "true")
    ),
    # Build provenance metadata embedded in official sonar-openai images.
    # Set via Docker ENV at image build time; informational only.
    "SONAR_BUILD_COMMIT": lambda: os.environ.get("SONAR_BUILD_COMMIT", "unknown"),
    "SONAR_BUILD_PIPELINE": lambda: os.environ.get("SONAR_BUILD_PIPELINE", "local"),
    "SONAR_BUILD_URL": lambda: os.environ.get("SONAR_BUILD_URL", ""),
    "SONAR_IMAGE_TAG": lambda: os.environ.get("SONAR_IMAGE_TAG", ""),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": env_with_choices("CMAKE_BUILD_TYPE", None, ["Debug", "Release", "RelWithDebInfo"]),
    # If set, sonar will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # Root directory for Sonar configuration files
    # Defaults to `~/.config/sonar` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how sonar finds its configuration files
    # during runtime, but also affects how sonar installs its configuration
    # files during **installation**.
    "SONAR_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "SONAR_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "sonar"),
        )
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for Sonar cache files
    # Defaults to `~/.cache/sonar` unless `XDG_CACHE_HOME` is set
    "SONAR_CACHE_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "SONAR_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "sonar"),
        )
    ),
    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    "SONAR_HOST_IP": lambda: os.getenv("SONAR_HOST_IP", ""),
    # used in distributed environment to manually set the communication port
    # Note: if SONAR_PORT is set, and some code asks for multiple ports, the
    # SONAR_PORT will be used as the first port, and the rest will be generated
    # by incrementing the SONAR_PORT value.
    "SONAR_PORT": get_sonar_port,
    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    "SONAR_RPC_BASE_PATH": lambda: os.getenv("SONAR_RPC_BASE_PATH", tempfile.gettempdir()),
    # If true, will load models from ModelScope instead of Hugging Face Hub.
    "SONAR_USE_MODELSCOPE": lambda: os.environ.get("SONAR_USE_MODELSCOPE", "False").strip().lower() in ("1", "true"),
    # If true, replace the Rust BPE backend that powers HF fast tokenizers
    # with the `fastokens` (https://github.com/crusoecloud/fastokens) shim.
    # Applies to HF fast tokenizers (`hf`, `deepseek_v32`, `deepseek_v4`, …).
    # Requires the `fastokens` Python package >= 0.3.5.
    "SONAR_USE_FASTOKENS": lambda: bool(int(os.getenv("SONAR_USE_FASTOKENS", "0"))),
    # Interval in seconds to log a warning message when the ring buffer is full
    "SONAR_RINGBUFFER_WARNING_INTERVAL": lambda: int(os.environ.get("SONAR_RINGBUFFER_WARNING_INTERVAL", "60")),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "SONAR_NCCL_SO_PATH": lambda: os.environ.get("SONAR_NCCL_SO_PATH", None),
    # when `SONAR_NCCL_SO_PATH` is not set, sonar will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # flag to control the chunk size (in MB) for sleeping memory allocations under ROCm
    "SONAR_ROCM_SLEEP_MEM_CHUNK_SIZE": lambda: int(os.environ.get("SONAR_ROCM_SLEEP_MEM_CHUNK_SIZE", "256")),
    # Feature flag to enable/disable Inductor standalone compile.
    # In torch <= 2.7 we ignore this flag; in torch >= 2.9 this is
    # enabled by default.
    "SONAR_USE_STANDALONE_COMPILE": lambda: os.environ.get("SONAR_USE_STANDALONE_COMPILE", "1") == "1",
    # Inductor's pre-grad passes don't do anything for Sonar.
    # The pre-grad passes get run even on cache-hit and negatively impact
    # sonar cold compile times by O(1s)
    # Can remove this after the following issue gets fixed
    # TODO(luka): maybe_inplace requires this
    # https://github.com/pytorch/pytorch/issues/174502
    "SONAR_ENABLE_PREGRAD_PASSES": lambda: os.environ.get("SONAR_ENABLE_PREGRAD_PASSES", "1") == "1",
    # Experimental: breakable cudagraph does not rely on torch.compile
    "SONAR_USE_BREAKABLE_CUDAGRAPH": lambda: os.environ.get("SONAR_USE_BREAKABLE_CUDAGRAPH", "0") == "1",
    # Debug pattern matching inside custom passes.
    # Should be set to the fx.Node name (e.g. 'getitem_34' or 'scaled_mm_3').
    "SONAR_PATTERN_MATCH_DEBUG": lambda: os.environ.get("SONAR_PATTERN_MATCH_DEBUG", None),
    # Dump fx graphs to the given directory.
    # It will override CompilationConfig.debug_dump_path if set.
    "SONAR_DEBUG_DUMP_PATH": lambda: os.environ.get("SONAR_DEBUG_DUMP_PATH", None),
    # Feature flag to enable/disable AOT compilation. This will ensure
    # compilation is done in warmup phase and the compilation will be
    # reused in subsequent calls.
    "SONAR_USE_AOT_COMPILE": use_aot_compile,
    # Feature flag to enable/disable bytecode in
    # TorchCompileWithNoGuardsWrapper.
    "SONAR_USE_BYTECODE_HOOK": lambda: bool(int(os.environ.get("SONAR_USE_BYTECODE_HOOK", "1"))),
    # Force sonar to always load AOT compiled models from disk. Failure
    # to load will result in a hard error when this is enabled.
    # Will be ignored when SONAR_USE_AOT_COMPILE is disabled.
    "SONAR_FORCE_AOT_LOAD": lambda: os.environ.get("SONAR_FORCE_AOT_LOAD", "0") == "1",
    # Enable loading compiled models directly from cached standalone compile artifacts
    # without re-splitting graph modules. This reduces overhead during model
    # loading by using reconstruct_serializable_fn_from_mega_artifact.
    "SONAR_USE_MEGA_AOT_ARTIFACT": use_mega_aot_artifact,
    # Compile with torch's compile-on-one-rank mode, which resolves the device
    # at run time instead of baking the compile-time index into the cached
    # code, so a compile cache can be reused on any device index (e.g. across
    # --device-ids). Off by default. Requires torch >= 2.14 on CUDA/ROCm or
    # XPU; not yet compatible with use_inductor_graph_partition or
    # debug_dump_path.
    "SONAR_COMPILE_ON_ONE_RANK": compile_on_one_rank,
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # Timeout in seconds for waiting for engine cores to become ready
    # during startup. Default is 600 seconds (10 minutes).
    "SONAR_ENGINE_READY_TIMEOUT_S": lambda: int(os.environ.get("SONAR_ENGINE_READY_TIMEOUT_S", "600")),
    # Maximum wall-clock seconds allowed for a single chat template render.
    # Set to 0 to disable the timeout.
    "SONAR_CHAT_TEMPLATE_RENDER_TIMEOUT": lambda: float(os.environ.get("SONAR_CHAT_TEMPLATE_RENDER_TIMEOUT", "30")),
    # Maximum template instructions (MiniJinja fuel) for rendering a
    # request-supplied chat template (--trust-request-chat-template) in the
    # Rust frontend; the server's own template has no limit. Set to 0 to
    # disable the limit.
    "SONAR_CHAT_TEMPLATE_FUEL": lambda: int(os.environ.get("SONAR_CHAT_TEMPLATE_FUEL", "500000")),
    # API key for Sonar API server
    "SONAR_API_KEY": lambda: os.environ.get("SONAR_API_KEY", None),
    # Whether to log responses from API Server for debugging
    "SONAR_DEBUG_LOG_API_SERVER_RESPONSE": lambda: (
        os.environ.get("SONAR_DEBUG_LOG_API_SERVER_RESPONSE", "False").lower() == "true"
    ),
    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID": lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY": lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL": lambda: os.environ.get("S3_ENDPOINT_URL", None),
    # Usage stats collection
    "SONAR_USAGE_STATS_SERVER": lambda: os.environ.get("SONAR_USAGE_STATS_SERVER", ""),
    "SONAR_NO_USAGE_STATS": lambda: os.environ.get("SONAR_NO_USAGE_STATS", "0") == "1",
    "SONAR_DO_NOT_TRACK": lambda: (
        (os.environ.get("SONAR_DO_NOT_TRACK", None) or os.environ.get("DO_NOT_TRACK", None) or "0") == "1"
    ),
    "SONAR_USAGE_SOURCE": lambda: os.environ.get("SONAR_USAGE_SOURCE", "production"),
    # Logging configuration
    # If set to 0, sonar will not configure logging
    # If set to 1, sonar will configure logging using the default configuration
    #    or the configuration file specified by SONAR_LOGGING_CONFIG_PATH
    "SONAR_CONFIGURE_LOGGING": lambda: bool(int(os.getenv("SONAR_CONFIGURE_LOGGING", "1"))),
    "SONAR_LOGGING_CONFIG_PATH": lambda: os.getenv("SONAR_LOGGING_CONFIG_PATH"),
    # this is used for configuring the default logging level
    "SONAR_LOGGING_LEVEL": lambda: os.getenv("SONAR_LOGGING_LEVEL", "INFO").upper(),
    # this is used for configuring the default logging stream
    "SONAR_LOGGING_STREAM": lambda: os.getenv("SONAR_LOGGING_STREAM", "ext://sys.stdout"),
    # if set, SONAR_LOGGING_PREFIX will be prepended to all log messages
    "SONAR_LOGGING_PREFIX": lambda: os.getenv("SONAR_LOGGING_PREFIX", ""),
    # Controls colored logging output. Options: "auto" (default, colors when terminal),
    # "1" (always use colors), "0" (never use colors)
    "SONAR_LOGGING_COLOR": lambda: os.getenv("SONAR_LOGGING_COLOR", "auto"),
    # Toggle the more verbose Sonar logging format (file/line info) instead
    # of the compact info format.
    "SONAR_LOGGING_VERBOSE": lambda: os.environ.get("SONAR_LOGGING_VERBOSE", "0").lower() in ("1", "true"),
    # Standard unix flag for disabling ANSI color codes
    "NO_COLOR": lambda: os.getenv("NO_COLOR", "0") != "0",
    # De-facto standard flag for forcing ANSI color codes (e.g. non-tty case)
    "FORCE_COLOR": lambda: os.getenv("FORCE_COLOR", "0") != "0",
    # If set, sonar will log stats at this interval in seconds
    # If not set, sonar will log stats every 10 seconds.
    "SONAR_LOG_STATS_INTERVAL": lambda: (
        val if (val := float(os.getenv("SONAR_LOG_STATS_INTERVAL", "10."))) > 0.0 else 10.0
    ),
    # If set to 1, sonar will collect and emit per-request level metrics.
    "SONAR_REQUEST_LEVEL_METRICS": lambda: bool(int(os.getenv("SONAR_REQUEST_LEVEL_METRICS", "1"))),
    # Trace function calls
    # If set to 1, sonar will trace function calls
    # Useful for debugging
    "SONAR_TRACE_FUNCTION": lambda: int(os.getenv("SONAR_TRACE_FUNCTION", "0")),
    # Whether to use the FlashInfer top-k / top-p sampler on CUDA. Enabled
    # by default when the hardware supports it — set to 0 to opt out
    # explicitly, which forces the PyTorch-native (Triton for bs>=8) path.
    "SONAR_USE_FLASHINFER_SAMPLER": lambda: (
        bool(int(os.environ["SONAR_USE_FLASHINFER_SAMPLER"])) if "SONAR_USE_FLASHINFER_SAMPLER" in os.environ else True
    ),
    # Pipeline stage partition strategy
    "SONAR_PP_LAYER_PARTITION": lambda: os.getenv("SONAR_PP_LAYER_PARTITION", None),
    # (CPU backend only) CPU key-value cache space.
    # default is None and will be set as 4 GB
    "SONAR_CPU_KVCACHE_SPACE": lambda: (
        int(os.getenv("SONAR_CPU_KVCACHE_SPACE", "0")) if "SONAR_CPU_KVCACHE_SPACE" in os.environ else None
    ),
    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "SONAR_CPU_OMP_THREADS_BIND": lambda: os.getenv("SONAR_CPU_OMP_THREADS_BIND", "auto"),
    # (CPU backend only) CPU cores not used by OMP threads .
    # Those CPU cores will not be used by OMP threads of a rank.
    "SONAR_CPU_NUM_OF_RESERVED_CPU": lambda: (
        int(os.getenv("SONAR_CPU_NUM_OF_RESERVED_CPU", "0")) if "SONAR_CPU_NUM_OF_RESERVED_CPU" in os.environ else None
    ),
    # (CPU backend only) whether Sonar is running in a CI environment.
    "SONAR_CPU_CI_ENV": lambda: bool(int(os.getenv("SONAR_CPU_CI_ENV", "0"))),
    # (CPU backend only) whether to enable attention spilt KV.
    "SONAR_CPU_ATTN_SPLIT_KV": lambda: bool(int(os.getenv("SONAR_CPU_ATTN_SPLIT_KV", "1"))),
    # (Zen CPU backend) eagerly prepack weights into ZenDNN blocked layout
    # at model load time. Eliminates per-inference layout conversion overhead.
    "SONAR_ZENTORCH_WEIGHT_PREPACK": lambda: bool(int(os.getenv("SONAR_ZENTORCH_WEIGHT_PREPACK", "1"))),
    # (CPU backend only) whether to use SGLang INT4 W4A8 kernels for AWQ, and on
    # Zen CPUs zentorch DA8W4 kernels for int4, and DA8W8 for grouped int8 MoE.
    "SONAR_CPU_INT4_W4A8": lambda: bool(int(os.getenv("SONAR_CPU_INT4_W4A8", "1"))),
    # Using a flag to control microbatch on XPU device, users on XPU device can
    # decide to disable it when they needed.
    "SONAR_XPU_PP_MICROBATCH": lambda: bool(int(os.getenv("SONAR_XPU_PP_MICROBATCH", "1"))),
    # When True, GroupCoordinator constructs its CPU/device subgroups via
    # ``torch.distributed.split_group(backend=...)``
    # and ``init_distributed_environment`` initializes the default PG with
    # mixed ``cpu:gloo,cuda:nccl`` backend + eager ``device_id`` binding.
    "SONAR_DISTRIBUTED_USE_SPLIT_GROUP": lambda: bool(int(os.getenv("SONAR_DISTRIBUTED_USE_SPLIT_GROUP", "0"))),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "SONAR_WORKER_MULTIPROC_METHOD": env_with_choices("SONAR_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork"]),
    # Path to the cache for storing downloaded assets
    "SONAR_ASSETS_CACHE": lambda: os.path.expanduser(
        os.getenv(
            "SONAR_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "sonar", "assets"),
        )
    ),
    # If the env var is set, we will clean model file in
    # this path $SONAR_ASSETS_CACHE/model_streamer/$model_name
    "SONAR_ASSETS_CACHE_MODEL_CLEAN": lambda: bool(int(os.getenv("SONAR_ASSETS_CACHE_MODEL_CLEAN", "0"))),
    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "SONAR_IMAGE_FETCH_TIMEOUT": lambda: int(os.getenv("SONAR_IMAGE_FETCH_TIMEOUT", "5")),
    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "SONAR_VIDEO_FETCH_TIMEOUT": lambda: int(os.getenv("SONAR_VIDEO_FETCH_TIMEOUT", "30")),
    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "SONAR_AUDIO_FETCH_TIMEOUT": lambda: int(os.getenv("SONAR_AUDIO_FETCH_TIMEOUT", "10")),
    # Directory for caching media downloads (images, video, audio fetched
    # from URLs during inference). Empty string disables caching.
    "SONAR_MEDIA_CACHE": lambda: os.getenv("SONAR_MEDIA_CACHE", ""),
    # Maximum cache size in MB. When exceeded, least-recently-used entries
    # are evicted. Default is 5120 (5 GB).
    "SONAR_MEDIA_CACHE_MAX_SIZE_MB": lambda: int(os.getenv("SONAR_MEDIA_CACHE_MAX_SIZE_MB", "5120")),
    # Time-to-live in hours for cached media files. Entries older than this
    # are evicted regardless of cache size. Default is 24 hours.
    "SONAR_MEDIA_CACHE_TTL_HOURS": lambda: float(os.getenv("SONAR_MEDIA_CACHE_TTL_HOURS", "24")),
    # Maximum number of retries for fetching media (images, audio, video)
    # from URLs. Each retry quadruples the timeout. Default is 3.
    "SONAR_MEDIA_FETCH_MAX_RETRIES": lambda: int(os.getenv("SONAR_MEDIA_FETCH_MAX_RETRIES", "3")),
    # Maximum size in MB for a single remote media download. The limit is
    # enforced while streaming the response body so oversized or infinite
    # responses cannot grow the API server heap without bound. Default is 256.
    "SONAR_MAX_MEDIA_DOWNLOAD_SIZE_MB": lambda: int(os.getenv("SONAR_MAX_MEDIA_DOWNLOAD_SIZE_MB", "256")),
    # Whether to allow HTTP redirects when fetching from media URLs.
    # Default to True
    "SONAR_MEDIA_URL_ALLOW_REDIRECTS": lambda: bool(int(os.getenv("SONAR_MEDIA_URL_ALLOW_REDIRECTS", "1"))),
    # Max number of workers for the thread pool handling
    # media bytes loading. Set to 1 to disable parallel processing.
    # Default is 8
    "SONAR_MEDIA_LOADING_THREAD_COUNT": lambda: int(os.getenv("SONAR_MEDIA_LOADING_THREAD_COUNT", "8")),
    # Maximum filesize in MB for a single audio file. Enforced on all
    # audio inputs (multimodal chat, speech-to-text uploads, data: URLs,
    # and local file:// paths). Files larger than this will be rejected
    # before decoding. Default is 25 MB.
    "SONAR_MAX_AUDIO_CLIP_FILESIZE_MB": lambda: int(os.getenv("SONAR_MAX_AUDIO_CLIP_FILESIZE_MB", "25")),
    # Maximum decoded audio duration in seconds.  Compressed audio files
    # (e.g. OPUS at very low bitrate) can expand into gigabytes of float32
    # PCM.  This limit is enforced *during* decoding so the memory is never
    # allocated.  Default is 600s (10 minutes).
    "SONAR_MAX_AUDIO_DECODE_DURATION_S": lambda: int(os.getenv("SONAR_MAX_AUDIO_DECODE_DURATION_S", "600")),
    # Maximum float32 PCM bytes that audio decoding may allocate.
    # Guards against sample-rate forgery where an attacker inflates
    # the header sample rate to bypass the duration guard while the
    # actual frame count still causes a multi-GiB allocation.
    # Default is 256 MiB (sufficient for 600s mono 48 kHz float32).
    "SONAR_MAX_AUDIO_DECODE_BYTES": lambda: int(os.getenv("SONAR_MAX_AUDIO_DECODE_BYTES", "268435456")),
    # Maximum bytes a client-supplied embedding payload (`prompt_embeds`,
    # `image_embeds`, `audio_embeds`, `video_embeds`) may allocate once it is
    # densified. A sparse tensor carries its own declared shape, so a payload
    # of a few hundred bytes can expand into hundreds of GiB. The limit is
    # checked before `to_dense()` so the memory is never allocated. Set to 0
    # to disable. Default is 2 GiB, which covers a 128K-token float32
    # embedding at hidden_size 4096.
    "SONAR_MAX_EMBED_DECODE_BYTES": lambda: int(os.getenv("SONAR_MAX_EMBED_DECODE_BYTES", "2147483648")),
    # Maximum number of worker threads used for STT preprocessing. The default
    # intentionally caps at 2 because that performed best in profiling.
    # https://github.com/vllm-project/vllm/pull/44612#issuecomment-4662757781
    "SONAR_MAX_AUDIO_PREPROCESS_WORKERS": lambda: int(
        os.getenv(
            "SONAR_MAX_AUDIO_PREPROCESS_WORKERS",
            str(max(1, min(os.cpu_count() or 1, 2))),
        )
    ),
    # Maximum decoded image size in pixels.  Small compressed images can
    # expand into gigabytes of raster memory.  This limit is enforced before
    # decoding so the memory is never allocated.  Default matches PIL's
    # built-in 2x decompression-bomb threshold (~179M pixels, ~680 MB RGB).
    "SONAR_MAX_IMAGE_PIXELS": lambda: int(os.getenv("SONAR_MAX_IMAGE_PIXELS", "178956970")),
    # Maximum aggregate decoded bytes across media in one request.
    # The default allows one image at the per-image pixel limit at 4 bytes
    # per pixel. Set to 0 to disable.
    "SONAR_MAX_MEDIA_REQUEST_DECODE_BYTES": lambda: int(os.getenv("SONAR_MAX_MEDIA_REQUEST_DECODE_BYTES", "715827880")),
    "SONAR_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB": lambda: int(
        os.getenv("SONAR_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", "256")
    ),
    # Backend for Video IO — selects the frame-sampling algorithm.
    # - "opencv": uniform sampling.
    # - "opencv_dynamic": duration-aware dynamic sampling.
    #
    # Custom backend implementations can be registered
    # via `@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "SONAR_VIDEO_LOADER_BACKEND": lambda: os.getenv("SONAR_VIDEO_LOADER_BACKEND", "opencv"),
    # Media connector implementation.
    # - "http": Default connector that supports fetching media via HTTP.
    #
    # Custom implementations can be registered
    # via `@MEDIA_CONNECTOR_REGISTRY.register("my_custom_media_connector")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "SONAR_MEDIA_CONNECTOR": lambda: os.getenv("SONAR_MEDIA_CONNECTOR", "http"),
    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "SONAR_XLA_CACHE_PATH": lambda: os.path.expanduser(
        os.getenv(
            "SONAR_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "sonar", "xla_cache"),
        )
    ),
    # If set, assert on XLA recompilation after each execution step.
    "SONAR_XLA_CHECK_RECOMPILATION": lambda: bool(int(os.getenv("SONAR_XLA_CHECK_RECOMPILATION", "0"))),
    # Enable SPMD mode for TPU backend.
    "SONAR_XLA_USE_SPMD": lambda: bool(int(os.getenv("SONAR_XLA_USE_SPMD", "0"))),
    # Use the single-launch fused FP8 MQA-logits + top-k kernel for the sparse
    # MLA indexer prefill chunks it supports on SM100-family GPUs (see
    # use_fused_mqa_topk / fused_mqa_topk_chunk); other chunks take the dense
    # path. Default: on (set to 0 for the dense path).
    "SONAR_USE_FUSED_MQA_TOPK": lambda: bool(int(os.getenv("SONAR_USE_FUSED_MQA_TOPK", "1"))),
    # Maximum size (in MB) for logits tensor in sparse MLA indexer prefill chunks.
    # Bounds the [M, N] float32 logits tensor to prevent CUDA OOM.
    # Default: 512 MB
    "SONAR_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(os.getenv("SONAR_SPARSE_INDEXER_MAX_LOGITS_MB", "512")),
    # Per-chunk decode logits budget (MiB) for the SM100 sparse indexer;
    # 0 disables decode chunking.
    "SONAR_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB": lambda: int(
        os.getenv("SONAR_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB", "2048")
    ),
    # KV context length each adaptive-verification profiling request pretends to
    # carry, so the profiled step reads a realistic amount of cache.
    # Raise it for long-context deployments, where step cost is dominated by
    # attention over a much larger KV cache than the default assumes.
    # Clamped to max_model_len - query_len. Default: 8192 tokens
    "SONAR_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN": lambda: int(
        os.getenv("SONAR_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "8192")
    ),
    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "SONAR_KEEP_ALIVE_ON_ENGINE_DEATH": lambda: bool(int(os.getenv("SONAR_KEEP_ALIVE_ON_ENGINE_DEATH", "0"))),
    # If the env var SONAR_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set SONAR_ALLOW_LONG_MAX_MODEL_LEN=1.
    "SONAR_ALLOW_LONG_MAX_MODEL_LEN": lambda: (
        os.environ.get("SONAR_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower() in ("1", "true")
    ),
    "SONAR_TEST_FORCE_LOAD_FORMAT": lambda: os.getenv("SONAR_TEST_FORCE_LOAD_FORMAT", "dummy"),
    # Queue size for fastsafetensors ParallelLoader pipelined weight
    # loading. Peak load-time VRAM is roughly
    # model_weights + (1 + queue_size) * shard_size.
    # Default 0 preserves the non-pipelined memory footprint so this
    # change does not shrink the loadable-model envelope. Set to 1
    # (or higher) to overlap producing the next shard's device buffer
    # with the consumer copying the current shard into model params,
    # at the cost of `queue_size` extra shard-sized buffers resident
    # at peak during loading.
    "SONAR_FASTSAFETENSORS_QUEUE_SIZE": lambda: int(os.getenv("SONAR_FASTSAFETENSORS_QUEUE_SIZE", "0")),
    # Timeout in seconds for keeping HTTP connections alive in API server
    "SONAR_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(os.environ.get("SONAR_HTTP_TIMEOUT_KEEP_ALIVE", "5")),
    # Maximum request body size in bytes accepted by the HTTP API server; 0
    # removes the limit. Only enforced by the Rust frontend; the Python
    # frontend has no limit.
    "SONAR_HTTP_MAX_REQUEST_BODY_SIZE": lambda: int(
        os.environ.get("SONAR_HTTP_MAX_REQUEST_BODY_SIZE", str(32 * 1024 * 1024))
    ),
    # Maximum allowed value for the `n` sampling parameter (number of output
    # sequences per request). Limits resource consumption to prevent
    # denial-of-service via excessively large fan-out. Default: 16384.
    "SONAR_MAX_N_SEQUENCES": lambda: int(os.environ.get("SONAR_MAX_N_SEQUENCES", "16384")),
    # Maximum number of prompts allowed in a single /v1/completions request
    # when the prompt field is a list. Prevents unbounded fan-out of engine
    # requests from a single API call. Default: 1024.
    "SONAR_MAX_COMPLETION_PROMPTS": lambda: int(os.environ.get("SONAR_MAX_COMPLETION_PROMPTS", "1024")),
    # Maximum number of stop strings allowed in a single request.
    "SONAR_MAX_STOP_STRINGS": lambda: int(os.environ.get("SONAR_MAX_STOP_STRINGS", "4")),
    # Maximum number of bad-word token sequences generated per request.
    "SONAR_MAX_NUM_BAD_WORDS": lambda: int(os.environ.get("SONAR_MAX_NUM_BAD_WORDS", "128")),
    # Maximum total number of bad-word tokens (summed across all bad words)
    # allowed per request. Bounds the per-request GPU buffer width.
    "SONAR_MAX_BAD_WORDS_TOTAL_TOKENS": lambda: int(os.environ.get("SONAR_MAX_BAD_WORDS_TOTAL_TOKENS", "1024")),
    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "SONAR_PLUGINS": lambda: None if "SONAR_PLUGINS" not in os.environ else os.environ["SONAR_PLUGINS"].split(","),
    # a local directory to look in for unrecognized LoRA adapters.
    # only works if plugins are enabled and
    # SONAR_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    "SONAR_LORA_RESOLVER_CACHE_DIR": lambda: os.getenv("SONAR_LORA_RESOLVER_CACHE_DIR", None),
    # A remote HF repo(s) containing one or more LoRA adapters, which
    # may be downloaded and leveraged as needed. Only works if plugins
    # are enabled and SONAR_ALLOW_RUNTIME_LORA_UPDATING is enabled.
    # Values should be comma separated.
    "SONAR_LORA_RESOLVER_HF_REPO_LIST": lambda: os.getenv("SONAR_LORA_RESOLVER_HF_REPO_LIST", None),
    # If set, Sonar will use Triton implementations of AWQ.
    "SONAR_USE_TRITON_AWQ": lambda: bool(int(os.getenv("SONAR_USE_TRITON_AWQ", "0"))),
    # If set, monkey-patch triton.runtime.autotuner.Autotuner.run to skip
    # benchmarking and select the first valid config (walking past invalid
    # ones). Used to eliminate autotuning variability when measuring kernel
    # performance and applied before running any kernel.
    "SONAR_TRITON_FORCE_FIRST_CONFIG": lambda: (
        os.environ.get("SONAR_TRITON_FORCE_FIRST_CONFIG", "0").strip().lower() in ("1", "true")
    ),
    # Maximum compiler threads per worker for registered CUDA Triton JIT warmup.
    # Set to 1 for serial warmup. Does not affect runtime JIT or autotuning.
    "SONAR_TRITON_JIT_WARMUP_NUM_THREADS": lambda: int(os.getenv("SONAR_TRITON_JIT_WARMUP_NUM_THREADS", "4")),
    # If set, allow loading or unloading lora adapters in runtime,
    "SONAR_ALLOW_RUNTIME_LORA_UPDATING": lambda: (
        os.environ.get("SONAR_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower() in ("1", "true")
    ),
    # We assume drivers can report p2p status correctly.
    # If the program hangs when using custom allreduce,
    # potantially caused by a bug in the driver (535 series),
    # if might be helpful to set SONAR_SKIP_P2P_CHECK=0
    # so that Sonar can verify if p2p is actually working.
    # See https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    "SONAR_SKIP_P2P_CHECK": lambda: os.getenv("SONAR_SKIP_P2P_CHECK", "1") == "1",
    # How a PP stage sends its intermediate tensors to the next stage.
    # "copy_engine" uses CUDA copy engines instead of NCCL kernels; it needs
    # TP=1 and NVLink-connected PP GPUs on one node, and fails startup without.
    # "auto" uses copy engines when every PP rank can, and NCCL otherwise.
    "SONAR_PP_TRANSPORT": env_with_choices("SONAR_PP_TRANSPORT", "auto", ["auto", "nccl", "copy_engine"]),
    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "SONAR_DISABLED_KERNELS": lambda: (
        [] if "SONAR_DISABLED_KERNELS" not in os.environ else os.environ["SONAR_DISABLED_KERNELS"].split(",")
    ),
    # Selects hw-agnostic layers for HF transformer backend
    "SONAR_USE_HW_AGNOSTIC": lambda: os.getenv("SONAR_USE_HW_AGNOSTIC", "False").lower() in ("true", "1"),
    "SONAR_ENABLE_FLA_PACKED_RECURRENT_DECODE": lambda: bool(
        int(os.getenv("SONAR_ENABLE_FLA_PACKED_RECURRENT_DECODE", "1"))
    ),
    # Select the GDN MTP decode implementation. "cuda" uses the fused decode
    # kernel where supported and falls back to "triton" otherwise; setting it
    # explicitly to "cuda" raises when unsupported.
    "SONAR_GDN_DECODE_KERNEL": env_with_choices(
        "SONAR_GDN_DECODE_KERNEL",
        "cuda",
        ["cuda", "triton"],
        case_sensitive=False,
    ),
    # Disable pynccl (using torch.distributed instead)
    "SONAR_DISABLE_PYNCCL": lambda: os.getenv("SONAR_DISABLE_PYNCCL", "False").lower() in ("true", "1"),
    # Optional: enable external Oink custom ops (e.g., Blackwell RMSNorm).
    # Disabled by default.
    "SONAR_USE_OINK_OPS": lambda: os.getenv("SONAR_USE_OINK_OPS", "False").lower() in ("true", "1"),
    # Set to 1 to dequantize MXFP4 weights to BF16 once at load time and run as
    # a BF16 checkpoint (no per-step weight dequant). This improves emulation
    # latency at the cost of additional device memory. Default off.
    "SONAR_MXFP4_EMULATION_DEQUANT_AT_LOAD": lambda: (
        os.getenv("SONAR_MXFP4_EMULATION_DEQUANT_AT_LOAD", "False").lower() in ("true", "1")
    ),
    # Keep TRTLLM MXFP4 expert scales compressed and decode into shared scratch.
    "SONAR_MXFP4_COMPRESS_SCALES": lambda: bool(int(os.getenv("SONAR_MXFP4_COMPRESS_SCALES", "0"))),
    # On hardware without a native MXFP8 kernel (e.g. ROCm gfx942 / MI300), the
    # MXFP8 emulation path dequantizes weights MXFP8->BF16 once at load time and
    # runs as a BF16 checkpoint (no per-step dequant). Set to 0 to fall back to
    # per-step dequant: keeps the 1-byte MXFP8 weights (~half the weight memory)
    # at the cost of dequantizing every forward step (much slower). Default on.
    "SONAR_MXFP8_EMULATION_DEQUANT_AT_LOAD": lambda: (
        os.getenv("SONAR_MXFP8_EMULATION_DEQUANT_AT_LOAD", "True").lower() in ("true", "1")
    ),
    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    "SONAR_ROCM_USE_AITER": lambda: os.getenv("SONAR_ROCM_USE_AITER", "False").lower() in ("true", "1"),
    # Use AITER's CustomAllreduce as the custom-allreduce backend inside Sonar's
    # CudaCommunicator on ROCm. Also enables AITER AG/RS for DP communication.
    "SONAR_ROCM_USE_AITER_CUSTOM_AR": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_CUSTOM_AR", "True").lower() in ("true", "1")
    ),
    # Muse Spark on ROCm: dtype of the TP reduction before the post-expert norm.
    "SONAR_ROCM_MUSE_MOE_REDUCE_DTYPE": lambda: os.getenv("SONAR_ROCM_MUSE_MOE_REDUCE_DTYPE", "fp32").lower(),
    # Muse Spark on ROCm: online-quantize routed experts ("" or "mxfp4"; first
    # and last layer stay BF16).
    "SONAR_ROCM_MUSE_MOE_ONLINE_QUANT": lambda: os.getenv("SONAR_ROCM_MUSE_MOE_ONLINE_QUANT", "").lower(),
    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    # - use aiter tuned gemms for unquantized gemms
    "SONAR_ROCM_USE_AITER_LINEAR": lambda: os.getenv("SONAR_ROCM_USE_AITER_LINEAR", "True").lower() in ("true", "1"),
    "SONAR_ROCM_USE_AITER_LINEAR_HIPBMM": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_LINEAR_HIPBMM", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter moe ops.
    # By default is enabled.
    "SONAR_ROCM_USE_AITER_MOE": lambda: os.getenv("SONAR_ROCM_USE_AITER_MOE", "True").lower() in ("true", "1"),
    # Activation dtype for the Kimi-K3 SiTU MXFP4 MoE (AITER FlyDSL SiTUv2):
    # auto (= a4w4), a4w4, a8w4 or a16w4. Legacy 1/0 mean a4w4/a16w4.
    "SONAR_ROCM_USE_AITER_MOE_SITUV2": env_with_choices(
        "SONAR_ROCM_USE_AITER_MOE_SITUV2",
        "auto",
        ["auto", "a4w4", "a8w4", "a16w4", "0", "1"],
        case_sensitive=False,
    ),
    # Opt-in switch for a4w4 (FP4 activation) MoE on DeepSeek V4.1, AITER
    # MXFP4 backend. Default is a8w4 (FP8); set to "1" to enable a4w4
    # ("true" is not accepted -- only "0"/"1"). Raises if set for other
    # models.
    "SONAR_ROCM_USE_AITER_MOE_A4W4_DSV4": lambda: maybe_convert_bool(os.getenv("SONAR_ROCM_USE_AITER_MOE_A4W4_DSV4")),
    # MoE sorting dispatch policy for AITER fused MoE kernels.
    #   0 = auto (default): single-pass for small batches, multi-pass
    #       for large batches
    #   1 = always single-pass: one kernel launch, no workspace,
    #       may be preferred for low-concurrency decode workloads
    #   2 = always multi-pass: can be faster for MoE-heavy models
    #       (e.g., +2-5% on Qwen3-Next, +1.5% on DeepSeek-V3 at TP4,
    #       see PR #39177 for benchmarks)
    "SONAR_ROCM_AITER_MOE_DISPATCH_POLICY": lambda: int(os.getenv("SONAR_ROCM_AITER_MOE_DISPATCH_POLICY", "0")),
    # use aiter rms norm op if aiter ops are enabled.
    "SONAR_ROCM_USE_AITER_RMSNORM": lambda: os.getenv("SONAR_ROCM_USE_AITER_RMSNORM", "True").lower() in ("true", "1"),
    # Whether to use aiter mla ops.
    # By default is enabled.
    "SONAR_ROCM_USE_AITER_MLA": lambda: os.getenv("SONAR_ROCM_USE_AITER_MLA", "True").lower() in ("true", "1"),
    # Small-head (<16) AITER MLA decode kernel selection. Small head counts
    # (e.g. Kimi-K3: 12 heads/rank at TP8, 6 at TP16) can decode either through
    # the Gluon small-head kernel or through the padded persistent-scheduling
    # (PS) ASM kernel. "auto" (default) keeps Gluon for head counts that divide
    # 16 where a Gluon build exists (gfx950/CDNA4) and otherwise uses the padded
    # PS ASM decode; "gluon" forces the Gluon path wherever a build exists;
    # "asm" forces the padded PS ASM decode. On gfx942/CDNA3 there is no Gluon
    # build, so the ASM path is always used regardless of this setting.
    "SONAR_ROCM_AITER_MLA_ASM_PADDING": env_with_choices(
        "SONAR_ROCM_AITER_MLA_ASM_PADDING",
        "auto",
        ["auto", "gluon", "asm"],
        case_sensitive=False,
    ),
    # Whether to use aiter mha ops.
    # By default is enabled.
    "SONAR_ROCM_USE_AITER_MHA": lambda: os.getenv("SONAR_ROCM_USE_AITER_MHA", "True").lower() in ("true", "1"),
    # Whether sparse MLA prefill and decode run on aiter's Triton kernel, which
    # reads the KV cache as stored (bf16 or fp8, paged or flat). Used by the
    # ROCM_AITER_MLA_SPARSE backend (DeepSeek V3.2, GLM-5.x) and DeepSeek
    # V4 / V4.1. gfx950 only. By default is disabled.
    "SONAR_ROCM_USE_AITER_TRITON_SPARSE_MLA": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_TRITON_SPARSE_MLA", "False").lower() in ("true", "1")
    ),
    # Run the sparse MLA calls the FlyDSL kernel accepts (bf16 q, fp8_ds_mla
    # cache, no sinks: Kimi-K3.1 DSA) on it instead of Triton; others stay on
    # Triton. gfx950 only. Disabled by default.
    "SONAR_ROCM_USE_AITER_FLYDSL_SPARSE_MLA": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_FLYDSL_SPARSE_MLA", "False").lower() in ("true", "1")
    ),
    # Smallest query batch SONAR_ROCM_USE_AITER_FLYDSL_SPARSE_MLA sends to the
    # FlyDSL kernel; smaller ones (decode) stay on Triton, which is faster
    # there. 160 is the measured gfx950 crossover for 2048-key rows, H 8..16.
    "SONAR_ROCM_AITER_FLYDSL_SPARSE_MLA_MIN_TOKENS": lambda: int(
        os.getenv("SONAR_ROCM_AITER_FLYDSL_SPARSE_MLA_MIN_TOKENS", "160")
    ),
    # Whether to use aiter rope.
    # By default is disabled.
    "SONAR_ROCM_USE_AITER_TRITON_ROPE": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_TRITON_ROPE", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter triton fp8 bmm kernel for MLA kv_b_proj.
    # By default is disabled: it re-quantizes the projection weights online,
    # regardless of the precision the checkpoint asked for.
    "SONAR_ROCM_USE_AITER_FP8BMM": lambda: os.getenv("SONAR_ROCM_USE_AITER_FP8BMM", "False").lower() in ("true", "1"),
    # Whether to use aiter triton fp4 bmm kernel for MLA kv_b_proj.
    # By default is disabled, for the same reason as the fp8 variant above.
    "SONAR_ROCM_USE_AITER_FP4BMM": lambda: os.getenv("SONAR_ROCM_USE_AITER_FP4BMM", "False").lower() in ("true", "1"),
    # Use AITER triton unified attention for V1 attention
    "SONAR_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter fusion shared experts ops.
    # By default is disabled.
    "SONAR_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter triton kernels for gemm ops.
    # By default is enabled.
    "SONAR_ROCM_USE_AITER_TRITON_GEMM": lambda: (
        os.getenv("SONAR_ROCM_USE_AITER_TRITON_GEMM", "True").lower() in ("true", "1")
    ),
    # use rocm skinny gemms
    "SONAR_ROCM_USE_SKINNY_GEMM": lambda: os.getenv("SONAR_ROCM_USE_SKINNY_GEMM", "True").lower() in ("true", "1"),
    # Pad the fp8 weights to 256 bytes for ROCm
    "SONAR_ROCM_FP8_PADDING": lambda: bool(int(os.getenv("SONAR_ROCM_FP8_PADDING", "1"))),
    # Pad the weights for the moe kernel
    "SONAR_ROCM_MOE_PADDING": lambda: bool(int(os.getenv("SONAR_ROCM_MOE_PADDING", "1"))),
    # Whether to use the shuffled kv cache layout
    "SONAR_ROCM_SHUFFLE_KV_CACHE_LAYOUT": lambda: (
        os.getenv("SONAR_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "False").lower() in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Choice of quantization level: FP, INT8, INT6, INT4, INT3 or NONE
    # Recommended for large models to get allreduce
    "SONAR_ROCM_QUICK_REDUCE_QUANTIZATION": env_with_choices(
        "SONAR_ROCM_QUICK_REDUCE_QUANTIZATION",
        "NONE",
        ["FP", "INT8", "INT6", "INT4", "INT3", "NONE"],
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Due to the lack of the bfloat16 asm instruction, bfloat16
    # kernels are slower than fp16,
    # If environment variable is set to 1, the input is converted to fp16
    "SONAR_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": lambda: (
        os.getenv("SONAR_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower() in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the maximum allowed number of data bytes(MB) for custom quick
    # allreduce communication.
    # Default: 2048 MB.
    # Data exceeding this size will use either custom allreduce or RCCL
    # communication.
    "SONAR_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": lambda: maybe_convert_int(
        os.environ.get("SONAR_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the minimum allowed number of data bytes(MB) required to use
    # custom quick allreduce communication.
    # If unset, use the built-in threshold table.
    "SONAR_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB": lambda: maybe_convert_int(
        os.environ.get("SONAR_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", None)
    ),
    # Controls the minimum tensor size (KB, where 1 KB = 1024 bytes) required
    # to use the configured QuickReduce codec. Smaller tensors use FP
    # QuickReduce. This does not affect QuickReduce eligibility.
    "SONAR_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB": lambda: maybe_convert_int(
        os.environ.get("SONAR_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", None)
    ),
    # Enable parallel decode top-16 selection for the MiniMax-M3.1 NVFP4
    # indexer on ROCm gfx950 (MI355X). Set to 1 to enable; disabled by default.
    # Uses runtime shape metadata for mixed batches. FP8 indexers (including
    # mono decode) bypass this path. Requires an image containing this kernel;
    # select the recipe's parallel_topk feature or set this variable explicitly.
    "SONAR_ROCM_PARALLEL_INDEX_TOPK": lambda: os.getenv("SONAR_ROCM_PARALLEL_INDEX_TOPK", "0") == "1",
    # If set, enable multiprocessing in LLM for the V1 code path.
    "SONAR_ENABLE_V1_MULTIPROCESSING": lambda: bool(int(os.getenv("SONAR_ENABLE_V1_MULTIPROCESSING", "1"))),
    "SONAR_LOG_BATCHSIZE_INTERVAL": lambda: float(os.getenv("SONAR_LOG_BATCHSIZE_INTERVAL", "-1")),
    "SONAR_DISABLE_COMPILE_CACHE": disable_compile_cache,
    # If set to "0", disable LayerName opaque type for layer_name
    # parameters in custom ops.  Defaults to enabled on torch >= 2.11.
    "SONAR_USE_LAYERNAME": lambda: bool(int(os.getenv("SONAR_USE_LAYERNAME", "1"))),
    # If set, use the Rust frontend binary instead of the Python API server
    # process(es).
    "SONAR_USE_RUST_FRONTEND": lambda: bool(int(os.getenv("SONAR_USE_RUST_FRONTEND", "0"))),
    # If set, use the packaged Rust client for `sonar bench serve`.
    "SONAR_USE_RUST_BENCH": lambda: bool(int(os.getenv("SONAR_USE_RUST_BENCH", "0"))),
    # Path to the sonar-rs binary. Defaults to "auto" which discovers
    # the binary installed with the sonar package. Used when
    # SONAR_USE_RUST_FRONTEND=1 or SONAR_USE_RUST_BENCH=1.
    "SONAR_RUST_FRONTEND_PATH": lambda: _resolve_rust_cli_path(),
    # If set, sonar will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`, and default `--enable-scale-out` to on
    "SONAR_SERVER_DEV_MODE": lambda: bool(int(os.getenv("SONAR_SERVER_DEV_MODE", "0"))),
    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "SONAR_V1_OUTPUT_PROC_CHUNK_SIZE": lambda: int(os.getenv("SONAR_V1_OUTPUT_PROC_CHUNK_SIZE", "128")),
    # If set, Sonar will disable the MLA attention optimizations.
    "SONAR_MLA_DISABLE": lambda: bool(int(os.getenv("SONAR_MLA_DISABLE", "0"))),
    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as Sonar.
    "SONAR_RAY_PER_WORKER_GPUS": lambda: float(os.getenv("SONAR_RAY_PER_WORKER_GPUS", "1.0")),
    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "SONAR_RAY_BUNDLE_INDICES": lambda: os.getenv("SONAR_RAY_BUNDLE_INDICES", ""),
    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable SONAR_CUDART_SO_PATH.
    "SONAR_CUDART_SO_PATH": lambda: os.getenv("SONAR_CUDART_SO_PATH", None),
    # Rank of the process in the data parallel setting
    "SONAR_DP_RANK": lambda: int(os.getenv("SONAR_DP_RANK", "0")),
    # Rank of the process in the data parallel setting.
    # Defaults to SONAR_DP_RANK when not set.
    "SONAR_DP_RANK_LOCAL": lambda: int(os.getenv("SONAR_DP_RANK_LOCAL", sys.modules[__name__].SONAR_DP_RANK)),
    # World size of the data parallel setting
    "SONAR_DP_SIZE": lambda: int(os.getenv("SONAR_DP_SIZE", "1")),
    # IP address of the master node in the data parallel setting
    "SONAR_DP_MASTER_IP": lambda: os.getenv("SONAR_DP_MASTER_IP", "127.0.0.1"),
    # Port of the master node in the data parallel setting
    "SONAR_DP_MASTER_PORT": lambda: int(os.getenv("SONAR_DP_MASTER_PORT", "0")),
    # Randomize inputs during dummy runs when using Data Parallel
    "SONAR_RANDOMIZE_DP_DUMMY_INPUTS": lambda: os.environ.get("SONAR_RANDOMIZE_DP_DUMMY_INPUTS", "0") == "1",
    # Strategy to pack the data parallel ranks for Ray.
    # Available options:
    # - "fill":
    #   for DP master node, allocate exactly data-parallel-size-local DP ranks,
    #   for non-master nodes, allocate as many DP ranks as can fit;
    # - "strict":
    #   allocate exactly data-parallel-size-local DP ranks to each picked node;
    # - "span":
    #   Should be used only when a single DP rank requires multiple nodes.
    #   allocate one DP rank over as many nodes as required for set world_size;
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "SONAR_RAY_DP_PACK_STRATEGY": lambda: os.getenv("SONAR_RAY_DP_PACK_STRATEGY", "strict"),
    # Optional comma-separated list of node IPs that Ray data-parallel
    # placement groups may use. When set, create_dp_placement_groups only
    # considers these nodes (the DP master node is always included).
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "SONAR_RAY_DP_PLACEMENT_NODE_IPS": lambda: os.getenv("SONAR_RAY_DP_PLACEMENT_NODE_IPS", ""),
    # Comma-separated *additional* prefixes of env vars to copy from the
    # driver to Ray workers.  These are merged with the built-in defaults
    # defined in ``sonar.ray.ray_env`` (SONAR_, etc.).  Example: "MYLIB_,OTHER_"
    "SONAR_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": lambda: os.getenv("SONAR_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY", ""),
    # Comma-separated *additional* individual env var names to copy from
    # the driver to Ray workers.  Merged with the built-in defaults
    # defined in ``sonar.ray.ray_env`` (PYTHONHASHSEED).
    # Example: "MY_SECRET,MY_FLAG"
    "SONAR_RAY_EXTRA_ENV_VARS_TO_COPY": lambda: os.getenv("SONAR_RAY_EXTRA_ENV_VARS_TO_COPY", ""),
    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "SONAR_MODEL_REDIRECT_PATH": lambda: os.environ.get("SONAR_MODEL_REDIRECT_PATH", None),
    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "SONAR_MARLIN_USE_ATOMIC_ADD": lambda: os.environ.get("SONAR_MARLIN_USE_ATOMIC_ADD", "0") == "1",
    # The online quantization dtype for humming kernel
    "SONAR_HUMMING_ONLINE_QUANT_CONFIG": lambda: maybe_convert_json_str_or_file(
        os.environ.get("SONAR_HUMMING_ONLINE_QUANT_CONFIG", None)
    ),
    # The activation dtype config for humming kernel. Explicit schemas disable
    # fallback unless the config includes "allow_fallback": true.
    "SONAR_HUMMING_INPUT_QUANT_CONFIG": lambda: maybe_convert_json_str_or_file(
        os.environ.get("SONAR_HUMMING_INPUT_QUANT_CONFIG", None)
    ),
    # Whether to use fp16 accumulator mma
    "SONAR_HUMMING_USE_F16_ACCUM": lambda: maybe_convert_bool(os.environ.get("SONAR_HUMMING_USE_F16_ACCUM", "0")),
    # Whether to use indexed gemm for humming moe
    # if 1, force use indexed gemm
    # if 0, force use grouped gemm
    # if None, choose better gemm type automatically
    "SONAR_HUMMING_MOE_GEMM_TYPE": lambda: os.environ.get("SONAR_HUMMING_MOE_GEMM_TYPE", None),
    # Whether to use DeepEPLL kernels for NVFP4 quantization and dispatch method
    # only supported on Blackwell GPUs and with
    # https://github.com/deepseek-ai/DeepEP/pull/341
    "SONAR_DEEPEPLL_NVFP4_DISPATCH": lambda: bool(int(os.getenv("SONAR_DEEPEPLL_NVFP4_DISPATCH", "0"))),
    # Whether to turn on the outlines cache for V1
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "SONAR_V1_USE_OUTLINES_CACHE": lambda: os.environ.get("SONAR_V1_USE_OUTLINES_CACHE", "0") == "1",
    # Whether using Pathways
    "SONAR_TPU_USING_PATHWAYS": lambda: bool("proxy" in os.getenv("JAX_PLATFORMS", "").lower()),
    # Allow use of DeepGemm kernels for fused moe ops.
    "SONAR_USE_DEEP_GEMM": lambda: bool(int(os.getenv("SONAR_USE_DEEP_GEMM", "1"))),
    # Allow use of DeepGemm specifically for MoE fused ops (overrides only MoE).
    "SONAR_MOE_USE_DEEP_GEMM": lambda: bool(int(os.getenv("SONAR_MOE_USE_DEEP_GEMM", "1"))),
    # Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs.
    "SONAR_USE_DEEP_GEMM_E8M0": lambda: bool(int(os.getenv("SONAR_USE_DEEP_GEMM_E8M0", "1"))),
    # Whether to create TMA-aligned scale tensor when DeepGEMM is used.
    "SONAR_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": lambda: bool(
        int(os.getenv("SONAR_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", "1"))
    ),
    # Opt-in MLA DCP query replication: skip the decode query all-gather.
    "SONAR_DCP_Q_REPLICATE": lambda: bool(int(os.getenv("SONAR_DCP_Q_REPLICATE", "0"))),
    # DeepGemm JITs the kernels on-demand. The warmup attempts to make DeepGemm
    # JIT all the required kernels before model execution so there is no
    # JIT'ing in the hot-path. However, this warmup increases the engine
    # startup time by a couple of minutes.
    # Available options:
    #  - "skip"  : Skip warmup.
    #  - "full"  : Warmup deepgemm by running all possible gemm shapes the
    #   engine could encounter.
    #  - "relax" : Select gemm shapes to run based on some heuristics. The
    #   heuristic aims to have the same effect as running all possible gemm
    #   shapes, but provides no guarantees.
    "SONAR_DEEP_GEMM_WARMUP": env_with_choices(
        "SONAR_DEEP_GEMM_WARMUP",
        "relax",
        [
            "skip",
            "full",
            "relax",
        ],
    ),
    # Whether to use fused grouped_topk used for MoE expert selection.
    "SONAR_USE_FUSED_MOE_GROUPED_TOPK": lambda: bool(int(os.getenv("SONAR_USE_FUSED_MOE_GROUPED_TOPK", "1"))),
    # Skip cudagraph/DP padding tokens in the MoE path by forcing their expert
    # ids to -1 so the dispatch and experts drop them. Requires a MoE kernel that
    # treats topk_id == -1 as a skip sentinel. Supported CUDA grouped routers
    # fuse this into routing; the existing backend opt-in is still required.
    # Under DP, TRT-LLM block-FP8/MXFP8 monolithic MoE then takes Sonar's
    # routing instead of routing from logits; 0 restores the kernel's routing.
    "SONAR_MOE_SKIP_PADDING": lambda: bool(int(os.getenv("SONAR_MOE_SKIP_PADDING", "1"))),
    # Step5 SP dense/shared MLPs: opt into TP-sharded weights with AG/RS
    # instead of replicated weights and local-token computation.
    "SONAR_STEP5_SHARD_SP_SHARED_EXPERT": lambda: bool(int(os.getenv("SONAR_STEP5_SHARD_SP_SHARED_EXPERT", "0"))),
    # Kimi-K3 only. Under sequence-parallel MoE the dense and shared-expert MLPs
    # are replicated on every rank, so each rank streams the whole weight to
    # serve its own token shard. Shard them across TP instead: the MLP then
    # all-gathers the full token set, computes this rank's partial, and
    # reduce-scatters (which both sums across TP and restores the sequence
    # sharding). Trades weight bandwidth and resident memory for two collectives
    # per layer, so it only wins at low token counts: intended for decode
    # instances in a P/D disaggregated deployment, not for prefill or unified
    # serving.
    "SONAR_KIMI_K3_SHARD_SP_SHARED_EXPERT": lambda: bool(int(os.getenv("SONAR_KIMI_K3_SHARD_SP_SHARED_EXPERT", "0"))),
    # Kimi K3 only, and unrelated to the MoE flags above. Tap the pre-norm
    # AttnRes mixture, rather than the post-mixture sum, as the auxiliary
    # hidden state handed to a DFlash drafter. This changes the numerics the
    # speculator sees, so it is off by default while the effect is measured.
    "SONAR_KIMI_K3_AUX_ATTN_RES_STREAM": lambda: bool(int(os.getenv("SONAR_KIMI_K3_AUX_ATTN_RES_STREAM", "0"))),
    # Use the SM100 BF16 GEMM-AR kernel for eligible Kimi-K3 row-parallel
    # attention projections. All TP ranks must belong to one NVLink domain.
    "SONAR_KIMI_K3_GEMM_AR": lambda: bool(int(os.getenv("SONAR_KIMI_K3_GEMM_AR", "1"))),
    # Fuse eligible sequence-parallel row-parallel projections with their TP
    # reduce-scatter using the SM100 BF16/MXFP8 GEMM-RS kernel (Kimi-K3
    # attention/shared-expert projections, DeepSeek-V4.1 ``wo_b``). All TP
    # ranks must belong to one NVLink domain.
    "SONAR_ENABLE_GEMM_RS": lambda: bool(int(os.getenv("SONAR_ENABLE_GEMM_RS", "0"))),
    # Allow use of FlashInfer FP8 block-scale GEMM for linear layers.
    # This uses TensorRT-LLM kernels and requires SM90+ (Hopper).
    "SONAR_BLOCKSCALE_FP8_GEMM_FLASHINFER": lambda: bool(int(os.getenv("SONAR_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1"))),
    # Force b12x FP4 MoE to use BF16 activations.
    "SONAR_B12X_MOE_FP4_FORCE_A16": lambda: bool(int(os.getenv("SONAR_B12X_MOE_FP4_FORCE_A16", "0"))),
    "SONAR_DSV41_SM120_NATIVE": lambda: bool(int(os.getenv("SONAR_DSV41_SM120_NATIVE", "0"))),
    # Allow use of FlashInfer MxInt4 MoE kernels for fused moe ops.
    "SONAR_USE_FLASHINFER_MOE_INT4": lambda: bool(int(os.getenv("SONAR_USE_FLASHINFER_MOE_INT4", "0"))),
    # Split Rubin NVFP4 MoE across locality domains. Requires cuda-bindings >= 13.4,
    # a PyTorch build with torch.cuda.execute_on_streams, and FlashInfer with
    # locality-domain MoE support (see sonar/distributed/localization.py).
    "SONAR_USE_LOCALITY_DOMAINS": lambda: bool(int(os.getenv("SONAR_USE_LOCALITY_DOMAINS", "0"))),
    # Give each compute partition an equal share of all SMs, allowing remote SMs.
    "SONAR_MOE_LOCALIZE_BACKFILL": lambda: bool(int(os.getenv("SONAR_MOE_LOCALIZE_BACKFILL", "0"))),
    # FlashInfer CuteDSL NVFP4 MoE only: with locality domains, overlap the
    # output memset with FC1 on an auxiliary stream.
    "SONAR_FLASHINFER_CUTEDSL_MOE_LOCALIZE_MEMSET_OVERLAP": lambda: bool(
        int(os.getenv("SONAR_FLASHINFER_CUTEDSL_MOE_LOCALIZE_MEMSET_OVERLAP", "1"))
    ),
    # FlashInfer CuteDSL NVFP4 MoE only: with locality domains, release the
    # full-width weight storage after sharding, disabling the full-width
    # fallback. EPLB uses the registered shards instead of the originals.
    "SONAR_FLASHINFER_CUTEDSL_MOE_LOCALIZE_FREE_FULL_WEIGHTS": lambda: bool(
        int(os.getenv("SONAR_FLASHINFER_CUTEDSL_MOE_LOCALIZE_FREE_FULL_WEIGHTS", "0"))
    ),
    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "SONAR_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("SONAR_XGRAMMAR_CACHE_MB", "512")),
    # Maximum time in seconds allowed for regex compilation in structured
    # output backends (xgrammar, outlines). Prevents ReDoS attacks where
    # adversarial patterns cause exponential DFA state-space explosion.
    # Set to 0 to disable the timeout (not recommended in production).
    "SONAR_REGEX_COMPILATION_TIMEOUT_S": lambda: int(os.getenv("SONAR_REGEX_COMPILATION_TIMEOUT_S", "5")),
    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded. The unit is bytes.
    "SONAR_MSGPACK_ZERO_COPY_THRESHOLD": lambda: int(os.getenv("SONAR_MSGPACK_ZERO_COPY_THRESHOLD", "256")),
    # If set, allow insecure serialization using pickle.
    # This is useful for environments where it is deemed safe to use the
    # insecure method and it is needed for some reason.
    "SONAR_ALLOW_INSECURE_SERIALIZATION": lambda: bool(int(os.getenv("SONAR_ALLOW_INSECURE_SERIALIZATION", "0"))),
    # Temporary: skip adding random suffix to internal request IDs. May be
    # needed for KV connectors that match request IDs across instances.
    "SONAR_DISABLE_REQUEST_ID_RANDOMIZATION": lambda: bool(
        int(os.getenv("SONAR_DISABLE_REQUEST_ID_RANDOMIZATION", "0"))
    ),
    # IP address used for NIXL handshake between remote agents.
    "SONAR_NIXL_SIDE_CHANNEL_HOST": lambda: os.getenv("SONAR_NIXL_SIDE_CHANNEL_HOST", "localhost"),
    # Port used for NIXL handshake between remote agents.
    "SONAR_NIXL_SIDE_CHANNEL_PORT": lambda: int(os.getenv("SONAR_NIXL_SIDE_CHANNEL_PORT", "5600")),
    # Address the P2P KV-offload control socket binds to. Defaults to
    # ``localhost`` (loopback only); must be set to the node IP for
    # cross-host P2P so remote peers can reach the socket.
    "SONAR_P2P_SIDE_CHANNEL_HOST": lambda: os.getenv("SONAR_P2P_SIDE_CHANNEL_HOST", "localhost"),
    # Port the P2P KV-offload control socket binds to.
    "SONAR_P2P_SIDE_CHANNEL_PORT": lambda: int(os.getenv("SONAR_P2P_SIDE_CHANNEL_PORT", "5710")),
    # Maximum number of concurrent P2P peer sessions. Limits ZMQ socket
    # consumption to prevent denial-of-service via unbounded peer
    # creation from client-supplied kv_transfer_params. Default: 128.
    "SONAR_P2P_MAX_PEERS": lambda: int(os.getenv("SONAR_P2P_MAX_PEERS", "128")),
    # Seconds to wait for a P2P peer handshake to complete (connected
    # but not yet ready). Sessions exceeding this deadline are reaped.
    # Prevents never-responding peers from accumulating indefinitely.
    "SONAR_P2P_HANDSHAKE_TIMEOUT_S": lambda: int(os.getenv("SONAR_P2P_HANDSHAKE_TIMEOUT_S", "30")),
    # Seconds a P2P session may remain idle (no pending work, no
    # messages) before it is eligible for eviction. Idle eviction runs
    # only when remaining peer slots are within 8 of SONAR_P2P_MAX_PEERS.
    # Default: 300. Set to 0 to disable.
    "SONAR_P2P_IDLE_TIMEOUT_S": lambda: int(os.getenv("SONAR_P2P_IDLE_TIMEOUT_S", "300")),
    # Seconds a P2P WRITE may stay in progress before the source treats its
    # peer as faulted: no new WRITEs to it, and the session is torn down.
    # The stalled transfer and its pins are held until it stops. Default:
    # 20. Set to 0 to disable.
    "SONAR_P2P_TRANSFER_STALL_TIMEOUT_S": lambda: float(os.getenv("SONAR_P2P_TRANSFER_STALL_TIMEOUT_S", "20")),
    # Optional EC control address override; otherwise use get_ip().
    "SONAR_EC_SIDE_CHANNEL_HOST": lambda: os.getenv("SONAR_EC_SIDE_CHANNEL_HOST", ""),
    # EC control base port plus DP index. Zero requests an OS-assigned port.
    "SONAR_EC_SIDE_CHANNEL_PORT": lambda: int(os.getenv("SONAR_EC_SIDE_CHANNEL_PORT", "0")),
    # Per-request timeout (seconds) when a prefiller worker registers with
    # the Mooncake bootstrap server.
    "SONAR_MOONCAKE_CONNECTOR_TIMEOUT": lambda: float(os.getenv("SONAR_MOONCAKE_CONNECTOR_TIMEOUT", "30.0")),
    # Port used for Mooncake handshake between remote agents.
    "SONAR_MOONCAKE_BOOTSTRAP_PORT": lambda: int(os.getenv("SONAR_MOONCAKE_BOOTSTRAP_PORT", "8998")),
    # Log per-batch memory/disk tier breakdown on external GETs.
    "SONAR_MOONCAKE_STORE_TIER_LOG": lambda: (
        os.getenv("SONAR_MOONCAKE_STORE_TIER_LOG", "False").lower() in ("true", "1")
    ),
    # Probe prefix-lookup keys without a read lease and lease only the keys the
    # hit loads. Requires Mooncake >= 0.3.14.
    "SONAR_MOONCAKE_STORE_SELECTIVE_LEASE": lambda: (
        os.getenv("SONAR_MOONCAKE_STORE_SELECTIVE_LEASE", "False").lower() in ("true", "1")
    ),
    # Number of parallel KV-load receive threads per worker rank. Lets the
    # per-request control overhead (Python prep + master key lookup) of one
    # request overlap with the RDMA transfer of another, keeping the transfer
    # engine's queue pairs busy. Helps when that overhead is significant or
    # per-request batches are too small to saturate the link on their own.
    "SONAR_MOONCAKE_LOAD_RECV_THREADS": lambda: int(os.getenv("SONAR_MOONCAKE_LOAD_RECV_THREADS", "1")),
    # Fraction of the owner's DirectIO staging buffer to fill per GET batch.
    "SONAR_MOONCAKE_DISK_STAGING_USABLE_RATIO": lambda: float(
        os.getenv("SONAR_MOONCAKE_DISK_STAGING_USABLE_RATIO", "0.9")
    ),
    # Pin this rank to a specific owner segment ("host:port").
    "MOONCAKE_PREFERRED_SEGMENT": lambda: os.getenv("MOONCAKE_PREFERRED_SEGMENT"),
    # Override the hostname the rank registers as a Mooncake requester.
    "MOONCAKE_REQUESTER_LOCAL_HOSTNAME": lambda: os.getenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME"),
    # Override the directory for the FlashInfer autotune config cache.
    "SONAR_FLASHINFER_AUTOTUNE_CACHE_DIR": lambda: os.getenv("SONAR_FLASHINFER_AUTOTUNE_CACHE_DIR", None),
    # Comma-separated FlashInfer op names to exclude from autotuning, using
    # the heuristic fallback tactic instead. Unset: skip "fp4_gemm" when the
    # CuTe-DSL NVFP4 linear kernel is selected. Empty: skip nothing.
    "SONAR_FLASHINFER_AUTOTUNE_SKIP_OPS": lambda: (
        None
        if "SONAR_FLASHINFER_AUTOTUNE_SKIP_OPS" not in os.environ
        else [v.strip() for v in os.environ["SONAR_FLASHINFER_AUTOTUNE_SKIP_OPS"].split(",") if v.strip()]
    ),
    # Flashinfer fused allreduce backend.
    "SONAR_FLASHINFER_ALLREDUCE_BACKEND": env_with_choices(
        "SONAR_FLASHINFER_ALLREDUCE_BACKEND",
        "auto",
        ["auto", "trtllm", "mnnvl"],
    ),
    # Control the workspace buffer size for the FlashInfer backend.
    "SONAR_FLASHINFER_WORKSPACE_BUFFER_SIZE": lambda: int(
        os.getenv("SONAR_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(394 * 1024 * 1024))
    ),
    # Transmit MoE all-to-all combine (expert-output) payloads in FP8 instead
    # of BF16, halving NVLink traffic on the combine leg. Only takes effect
    # when the installed FlashInfer MoeAlltoAll kernel supports it.
    "SONAR_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE": lambda: bool(
        int(os.getenv("SONAR_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE", "0"))
    ),
    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE CUTLASS Kernel. This value is used to create a buffer for
    # the blockscale tensor of activations NVFP4 Quantization.
    # This is used to prevent the kernel from running out of memory.
    "SONAR_MAX_TOKENS_PER_EXPERT_FP4_MOE": lambda: int(os.getenv("SONAR_MAX_TOKENS_PER_EXPERT_FP4_MOE", "163840")),
    # MoE routing strategy selector.
    # See `RoutingSimulator.get_available_strategies()` # for available
    # strategies.
    # Custom routing strategies can be registered by
    # RoutingSimulator.register_strategy()
    # Note: custom strategies may not produce correct model outputs
    "SONAR_MOE_ROUTING_SIMULATION_STRATEGY": lambda: os.environ.get(
        "SONAR_MOE_ROUTING_SIMULATION_STRATEGY", ""
    ).lower(),
    # Regex timeout for use by the Sonar tool parsing plugins.
    "SONAR_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(os.getenv("SONAR_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")),
    # Enforce function parameter schemas in structural-tag based tool calling.
    "SONAR_ENFORCE_STRICT_TOOL_CALLING": lambda: (
        os.getenv("SONAR_ENFORCE_STRICT_TOOL_CALLING", "True").lower() in ("true", "1")
    ),
    # Constrain every `beam` output to its response template's format, also
    # without tool or structured output constraints.
    "SONAR_BEAM_ALWAYS_CONSTRAIN_OUTPUT": lambda: (
        os.getenv("SONAR_BEAM_ALWAYS_CONSTRAIN_OUTPUT", "True").lower() in ("true", "1")
    ),
    # Control the max chunk bytes (in MB) for the rpc message queue.
    # Object larger than this threshold will be broadcast to worker
    # processes via zmq.
    "SONAR_MQ_MAX_CHUNK_BYTES_MB": lambda: int(os.getenv("SONAR_MQ_MAX_CHUNK_BYTES_MB", "16")),
    # Timeout in seconds for execute_model RPC calls in multiprocessing
    # executor (only applies when TP > 1).
    "SONAR_EXECUTE_MODEL_TIMEOUT_SECONDS": lambda: int(os.getenv("SONAR_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")),
    # Timeout in seconds for engine and worker process shutdown
    "SONAR_WORKER_SHUTDOWN_TIMEOUT_SECONDS": lambda: int(os.getenv("SONAR_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")),
    # KV Cache layout used throughout sonar.
    # Some common values are:
    # - LBNHC
    # - LBHNC
    # Where N=num_states, H=num_heads and C=state_content. The default value will
    # leave the layout choice to the backend. Mind that backends may only
    # implement and support a subset of all possible layouts.
    # LHBNC hoists the head dim outside the block dim; backends must opt in via
    # AttentionBackend.supported_kv_cache_layouts().
    "SONAR_KV_CACHE_LAYOUT": env_with_choices(
        "SONAR_KV_CACHE_LAYOUT",
        None,
        ["LBNHC", "LBHNC", "LHBNC", "NHD", "HND", "BLHNC", "BLNHC", "BHLNC"],
    ),
    # SSM conv state layout used for Mamba models.
    # - SD: (state_len, dim) — dim contiguous (default)
    # - DS: (dim, state_len) — TP-sharded dim on dim1,
    #   consistent with SSM temporal state and LBHNC KV cache layout.
    "SONAR_SSM_CONV_STATE_LAYOUT": env_with_choices("SONAR_SSM_CONV_STATE_LAYOUT", None, ["SD", "DS"]),
    # Enable checking whether the generated logits contain NaNs,
    # indicating corrupted output. Useful for debugging low level bugs
    # or bad hardware but it may add compute overhead.
    "SONAR_COMPUTE_NANS_IN_LOGITS": lambda: bool(
        int(os.getenv("SONAR_COMPUTE_NANS_IN_LOGITS", "0")) or int(os.getenv("SONAR_RAISE_ON_LOGIT_NANS", "0"))
    ),
    # Raise an exception when generated logits contain NaNs. Enabling this
    # also enables the NaN computation required to detect them.
    "SONAR_RAISE_ON_LOGIT_NANS": lambda: bool(int(os.getenv("SONAR_RAISE_ON_LOGIT_NANS", "0"))),
    # Override NVIDIA Confidential Computing detection. Unset (default)
    # auto-detects via NVML; "1" forces it on, "0" forces it off. Gates the
    # Confidential Computing copy paths (see sonar.v1.conf_compute_utils).
    "SONAR_CONFIDENTIAL_COMPUTE": lambda: os.getenv("SONAR_CONFIDENTIAL_COMPUTE"),
    # Timeout (in seconds) for MooncakeConnector in PD disaggregated setup.
    "SONAR_MOONCAKE_ABORT_REQUEST_TIMEOUT": lambda: int(os.getenv("SONAR_MOONCAKE_ABORT_REQUEST_TIMEOUT", "480")),
    # If set, it means we pre-downloaded cubin files and flashinfer will
    # read the cubin files directly.
    "SONAR_HAS_FLASHINFER_CUBIN": lambda: bool(int(os.getenv("SONAR_HAS_FLASHINFER_CUBIN", "0"))),
    # Controls garbage collection during CUDA graph capture.
    # If set to 0 (default), enables GC freezing to speed up capture time.
    # If set to 1, allows GC to run during capture.
    "SONAR_ENABLE_CUDAGRAPH_GC": lambda: bool(int(os.getenv("SONAR_ENABLE_CUDAGRAPH_GC", "0"))),
    # Used to force set up loopback IP
    "SONAR_LOOPBACK_IP": lambda: os.getenv("SONAR_LOOPBACK_IP", ""),
    # Used to set the process name prefix for Sonar processes.
    # This is useful for debugging and monitoring purposes.
    # The default value is "SONAR".
    "SONAR_PROCESS_NAME_PREFIX": lambda: os.getenv("SONAR_PROCESS_NAME_PREFIX", "SONAR"),
    # Allow chunked local attention with hybrid kv cache manager.
    # Enabled by default; set to 0 to disable chunked local attention when using
    # the hybrid KV cache manager.
    "SONAR_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE": lambda: bool(
        int(os.getenv("SONAR_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "1"))
    ),
    # Enables support for the "store" option in the OpenAI Responses API.
    # When set to 1, Sonar's OpenAI server will retain the input and output
    # messages for those requests in memory. By default, this is disabled (0),
    # and the "store" option is ignored.
    # NOTE/WARNING:
    # 1. Messages are kept in memory only (not persisted to disk) and will be
    #    lost when the Sonar server shuts down.
    # 2. Enabling this option will cause a memory leak, as stored messages are
    #    never removed from memory until the server terminates.
    "SONAR_ENABLE_RESPONSES_API_STORE": lambda: bool(int(os.getenv("SONAR_ENABLE_RESPONSES_API_STORE", "0"))),
    # If set to 1, expose the Cohere Chat v2 API at ``POST /cohere/v2/chat``
    # and its render endpoint at ``POST /cohere/v2/chat/render``.
    # Default off
    "SONAR_ENABLE_COHERE_API": lambda: bool(int(os.getenv("SONAR_ENABLE_COHERE_API", "0"))),
    # If set, use the fp8 mfma in rocm paged attention.
    "SONAR_ROCM_FP8_MFMA_PAGE_ATTN": lambda: bool(int(os.getenv("SONAR_ROCM_FP8_MFMA_PAGE_ATTN", "0"))),
    # Whether to use pytorch symmetric memory for allreduce
    "SONAR_ALLREDUCE_USE_SYMM_MEM": lambda: bool(int(os.getenv("SONAR_ALLREDUCE_USE_SYMM_MEM", "1"))),
    # Whether to use FlashInfer allreduce
    "SONAR_ALLREDUCE_USE_FLASHINFER": lambda: bool(int(os.getenv("SONAR_ALLREDUCE_USE_FLASHINFER", "1"))),
    # Whether to use FlashInfer's single-node PCIe CUDA-IPC all-reduce.
    # This backend has a strict single-stream contract and is opt-in while its
    # integration is being qualified.
    "SONAR_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": lambda: bool(
        int(os.getenv("SONAR_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0"))
    ),
    # Use the fixed-order, single-lane SM120 TP4 staged PCIe pull backend.
    "SONAR_ALLREDUCE_USE_PCIE_PULL": lambda: bool(int(os.getenv("SONAR_ALLREDUCE_USE_PCIE_PULL", "0"))),
    # Host staging uses shared CPU DRAM and does not require GPU peer access.
    "SONAR_ALLREDUCE_PCIE_TRANSPORT": env_with_choices("SONAR_ALLREDUCE_PCIE_TRANSPORT", "paired", ["paired", "host"]),
    # Experimental: use this to enable MCP tool calling for non harmony models
    "SONAR_USE_EXPERIMENTAL_PARSER_CONTEXT": lambda: bool(int(os.getenv("SONAR_USE_EXPERIMENTAL_PARSER_CONTEXT", "0"))),
    # User override folder for tuned Triton-kernel configs. Shared by MoE,
    # Mamba SSU, and LoRA. Filenames are distinct so one folder can hold all.
    # Each component first checks this folder, then the configs shipped with
    # Sonar (if any). If no JSON matches, it uses a hard-coded heuristic.
    "SONAR_TUNED_CONFIG_FOLDER": lambda: os.getenv("SONAR_TUNED_CONFIG_FOLDER", None),
    # Opt-in persistence of the startup plan. When enabled, each worker
    # saves the memory-profiling result (the suggested --kv-cache-memory value
    # and the free-memory baseline) under SONAR_CACHE_ROOT/startup_plan/,
    # keyed by a hardware+config fingerprint, and later boots auto-apply it
    # -- skipping memory profiling -- when the fingerprint matches and
    # current free memory >= the recorded baseline.
    # See sonar/v1/worker/startup_plan.py.
    "SONAR_ENABLE_STARTUP_PLAN": lambda: bool(int(os.getenv("SONAR_ENABLE_STARTUP_PLAN", "0"))),
    # Valid values are container,code_interpreter,web_search_preview
    # ex SONAR_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=container,code_interpreter
    # If the server_label of your mcp tool is not in this list it will
    # be completely ignored.
    "SONAR_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": env_set_with_choices(
        "SONAR_GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
        default=[],
        choices=["container", "code_interpreter", "web_search_preview"],
    ),
    # Allows harmony instructions to be injected on system messages
    "SONAR_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": lambda: bool(
        int(os.getenv("SONAR_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "0"))
    ),
    # Pin the conversation start date injected into the Harmony system
    # message. When unset the current date is used, which introduces
    # non-determinism (different tokens -> different model behaviour at
    # temperature=0). Set to an ISO date string, e.g. "2023-09-12",
    # for reproducible inference or testing.
    "SONAR_SYSTEM_START_DATE": lambda: os.getenv("SONAR_SYSTEM_START_DATE", None),
    # Enable automatic retry when tool call JSON parsing fails
    # If enabled, returns an error message to the model to retry
    # If disabled (default), raises an exception and fails the request
    "SONAR_TOOL_JSON_ERROR_AUTOMATIC_RETRY": lambda: bool(int(os.getenv("SONAR_TOOL_JSON_ERROR_AUTOMATIC_RETRY", "0"))),
    # Add optional custom scopes for profiling, disable to avoid overheads
    "SONAR_CUSTOM_SCOPES_FOR_PROFILING": lambda: bool(int(os.getenv("SONAR_CUSTOM_SCOPES_FOR_PROFILING", "0"))),
    # Add optional nvtx scopes for profiling, disable to avoid overheads
    "SONAR_NVTX_SCOPES_FOR_PROFILING": lambda: bool(int(os.getenv("SONAR_NVTX_SCOPES_FOR_PROFILING", "0"))),
    # Represent block hashes in KV cache events as 64-bit integers instead of
    # raw bytes. Defaults to True for backward compatibility.
    "SONAR_KV_EVENTS_USE_INT_BLOCK_HASHES": lambda: bool(int(os.getenv("SONAR_KV_EVENTS_USE_INT_BLOCK_HASHES", "1"))),
    # Name of the shared memory buffer used for object storage.
    # Only effective when mm_config.mm_processor_cache_type == "shm".
    # Automatically generates a unique UUID-based name per process tree
    # if not explicitly set.
    "SONAR_OBJECT_STORAGE_SHM_BUFFER_NAME": get_env_or_set_default(
        "SONAR_OBJECT_STORAGE_SHM_BUFFER_NAME",
        _generate_shm_name,
    ),
    # The size in MB of the buffers (NVL and RDMA) used by DeepEP
    "SONAR_DEEPEP_BUFFER_SIZE_MB": lambda: int(os.getenv("SONAR_DEEPEP_BUFFER_SIZE_MB", "1024")),
    # Force DeepEP to use intranode kernel for inter-node communication in
    # high throughput mode. This is useful archive higher prefill throughput
    # on system supports multi-node nvlink (e.g GB200).
    "SONAR_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE": lambda: bool(
        int(os.getenv("SONAR_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE", "0"))
    ),
    # MoRI EPv2 for --all2all-backend mori. Unset prefers EPv2 and falls back to
    # EPv1 where EPv2 is unsupported; 1 forces EPv2; 0 selects EPv1.
    "SONAR_MORI_EP_V2": lambda: maybe_convert_bool(os.getenv("SONAR_MORI_EP_V2")),
    # With --all2all-backend mori on MoRI EPv1 across nodes, use InterNodeV1LL
    # when the per-rank token budget is at most this, else InterNodeV1.
    "SONAR_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD": lambda: int(
        os.getenv("SONAR_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", "256")
    ),
    # Allow DeepEP to use MNNVL (multi-node nvlink) for internode_ll kernel,
    # turn this for better latency on GB200 like system
    "SONAR_DEEPEP_LOW_LATENCY_USE_MNNVL": lambda: bool(int(os.getenv("SONAR_DEEPEP_LOW_LATENCY_USE_MNNVL", "0"))),
    # DeepEP v2: use NVLink+RDMA hybrid communication across NVLink domains.
    #   * hybrid mode = multi-node RDMA
    #   * non hybrid mode = nvlink
    # Set to 1 (the default) to allow DeepEP to autoselect hybrid vs non-hybrid.
    # Set to 0 to disable hybrid mode.
    "SONAR_DEEPEP_V2_ALLOW_HYBRID_MODE": lambda: bool(int(os.getenv("SONAR_DEEPEP_V2_ALLOW_HYBRID_MODE", "1"))),
    # DeepEP v2: use fewer SMs at slight throughput cost
    "SONAR_DEEPEP_V2_PREFER_OVERLAP": lambda: bool(int(os.getenv("SONAR_DEEPEP_V2_PREFER_OVERLAP", "0"))),
    # DeepEP v2: trade precision for transfer size in combine
    "SONAR_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION": lambda: bool(
        int(os.getenv("SONAR_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION", "0"))
    ),
    # The number of SMs/CUs to allocate for communication kernels when
    # running DBO; the rest will be allocated to compute.
    # Default: 20 on CUDA (SMs), 64 on ROCm (CUs).
    "SONAR_DBO_COMM_SMS": lambda: int(
        os.getenv(
            "SONAR_DBO_COMM_SMS",
            "64"
            if hasattr(__import__("torch").version, "hip") and __import__("torch").version.hip is not None
            else "20",
        )
    ),
    # Enable max_autotune & coordinate_descent_tuning in inductor_config
    # to compile static shapes passed from compile_sizes in compilation_config
    # If set to 1, enable max_autotune; By default, this is enabled (1)
    "SONAR_ENABLE_INDUCTOR_MAX_AUTOTUNE": lambda: bool(int(os.getenv("SONAR_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"))),
    # If set to 1, enable coordinate_descent_tuning;
    # By default, this is enabled (1)
    "SONAR_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING": lambda: bool(
        int(os.getenv("SONAR_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "1"))
    ),
    # Flag to enable NCCL symmetric memory allocation and registration
    "SONAR_USE_NCCL_SYMM_MEM": lambda: bool(int(os.getenv("SONAR_USE_NCCL_SYMM_MEM", "0"))),
    # NCCL header path
    "SONAR_NCCL_INCLUDE_PATH": lambda: os.environ.get("SONAR_NCCL_INCLUDE_PATH", None),
    # GC debug config
    # - SONAR_GC_DEBUG=0: disable GC debugger
    # - SONAR_GC_DEBUG=1: enable GC debugger with gc.collect elapsed times
    # - SONAR_GC_DEBUG='{"top_objects":5}': enable GC debugger with
    #                                      top 5 collected objects
    "SONAR_GC_DEBUG": lambda: os.getenv("SONAR_GC_DEBUG", ""),
    # Debug workspace allocations.
    # logging of workspace resize operations.
    "SONAR_DEBUG_WORKSPACE": lambda: bool(int(os.getenv("SONAR_DEBUG_WORKSPACE", "0"))),
    # Disables parallel execution of shared_experts via separate cuda stream
    "SONAR_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(int(os.getenv("SONAR_DISABLE_SHARED_EXPERTS_STREAM", "0"))),
    # Emergency rollback for the DeepSeek-V4 NVIDIA MegaMoE path. By default,
    # DeepGEMM computes replicated FP8 shared experts in the same persistent
    # SM100 kernel as the routed FP4 experts.
    "SONAR_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION": lambda: bool(
        int(os.getenv("SONAR_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION", "0"))
    ),
    # Size cap for the lm_head skinny-GEMM (wvSplitK) weight guard: weights
    # larger than this skip the skinny path at 1 < n <= 5 (see
    # sonar.model_executor.layers.utils.lmhead_skinny_guard_permits).
    # wvSplitK's cost at n activation rows scales as ~n passes over the
    # weight, so once the weight exceeds the last-level (memory-side)
    # cache those extra passes are HBM round trips. 0 (the default)
    # derives the cap from that cache size once at first use (256 MiB on
    # MI355X, 128 MiB on MI300X-class parts), with a 256 MiB fallback
    # where the size cannot be read — see the guard for how it's read.
    # The portable device-properties API cannot be used: it exposes only
    # the far smaller L2 (4 MiB on gfx950), which would disable the
    # fastpath on every real lm_head shard. Set a value explicitly to
    # override the hardware-derived cap; set it very large (e.g. 2**60)
    # to disable the guard.
    "SONAR_SKINNY_GEMM_MAX_WEIGHT_BYTES": lambda: int(os.getenv("SONAR_SKINNY_GEMM_MAX_WEIGHT_BYTES", "0")),
    # Limits when we run shared_experts in a separate stream.
    # We found out that for large batch sizes, the separate stream
    # execution is not beneficial (most likely because of the input clone)
    # TODO(alexm-redhat): Tune to be more dynamic based on GPU type
    "SONAR_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": lambda: int(
        int(os.getenv("SONAR_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 256))
    ),
    # Token-count cutoff for multi-stream overlap of the attention input
    # GEMM with auxiliary GEMMs (e.g. fused_wqa_wkv overlapped with indexer
    # weights / kv-score projections in DeepSeek-V4). At or below this many
    # tokens the FP8 main GEMM has idle SMs to share with the bf16 aux GEMMs
    # and overlap is a 5-45% win; above it the FP8 GEMM saturates the device
    # and the cross-stream sync becomes pure overhead. Set to 0 to disable
    # the multi-stream path entirely. See #PR 41526 for the empirical result
    # for the default value of 1024 tokens.
    "SONAR_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": lambda: int(
        os.getenv("SONAR_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", "1024")
    ),
    # Cohere2 MoE runs the MLP/MoE of its parallel blocks on a side CUDA stream,
    # overlapping it with attention, for batches of at most this many tokens.
    # Larger batches and piecewise CUDA graphs (prefill / mixed batches) run
    # the MLP inline. Set to 0 to disable the dual stream.
    "SONAR_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD": lambda: int(
        os.getenv("SONAR_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD", "256")
    ),
    # Format for saving torch.compile cache artifacts
    # - "binary": saves as binary file
    #     Safe for multiple sonar serve processes accessing the same torch compile cache.
    # - "unpacked": saves as directory structure (for inspection/debugging)
    #     NOT multiprocess safe - race conditions may occur with multiple processes.
    #     Allows viewing and setting breakpoints in Inductor's code output files.
    "SONAR_COMPILE_CACHE_SAVE_FORMAT": env_with_choices(
        "SONAR_COMPILE_CACHE_SAVE_FORMAT", "binary", ["binary", "unpacked"]
    ),
    # Log model inspection after loading.
    # If enabled, logs a transformers-style hierarchical view of the model
    # with quantization methods and attention backends.
    "SONAR_LOG_MODEL_INSPECTION": lambda: bool(int(os.getenv("SONAR_LOG_MODEL_INSPECTION", "0"))),
    # Debug logging for --enable-mfu-metrics
    "SONAR_DEBUG_MFU_METRICS": lambda: bool(int(os.getenv("SONAR_DEBUG_MFU_METRICS", "0"))),
    # Disable using pytorch's pin memory for CPU offloading.
    "SONAR_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": lambda: bool(
        int(os.getenv("SONAR_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "0"))
    ),
    # Disable using UVA (Unified Virtual Addressing) for CPU offloading.
    "SONAR_WEIGHT_OFFLOADING_DISABLE_UVA": lambda: bool(int(os.getenv("SONAR_WEIGHT_OFFLOADING_DISABLE_UVA", "0"))),
    # How weight updates reload the model: "layerwise" rebuilds each layer
    # from the checkpoint tensors; "direct" writes them into the live
    # tensors and reruns post-load processing in place.
    "SONAR_RELOAD_MODE": env_with_choices("SONAR_RELOAD_MODE", "layerwise", ["layerwise", "direct"]),
    # Max descriptors per CPU-KV-offload batch-memcpy call. 0 = platform default
    # (ROCm chunks at 8192, since hipMemcpyBatchAsync faults above that on
    # rocm 7.14/7.15 when batch copy is optimized; CUDA uncapped). Set >0 to override.
    "SONAR_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS": lambda: int(os.getenv("SONAR_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS", "0")),
    # On WSL2 with a compatible kernel (>= 4.19.121), pinned memory is
    # supported but disabled by default due to a small performance regression.
    # Set to 1 when pinned memory or UVA is required (e.g. CPU offloading
    # or v2 model runner).
    "SONAR_WSL2_ENABLE_PIN_MEMORY": lambda: bool(int(os.getenv("SONAR_WSL2_ENABLE_PIN_MEMORY", "0"))),
    # Disable logging of Sonar logo at server startup time.
    "SONAR_DISABLE_LOG_LOGO": lambda: bool(int(os.getenv("SONAR_DISABLE_LOG_LOGO", "0"))),
    # Kill switch for the sm89 (Ada, e.g. RTX 4090) DeepSeek sparse attention
    # kernels. Skips the SM89_MLA_SPARSE backend and the sm89 indexer logits
    # paths even if the kernels are compiled in.
    "SONAR_DISABLE_SM89_DSA": lambda: bool(int(os.getenv("SONAR_DISABLE_SM89_DSA", "0"))),
    # Disable PDL for LoRA, as enabling PDL with LoRA on SM100 causes
    # Triton compilation to fail.
    "SONAR_LORA_DISABLE_PDL": lambda: bool(int(os.getenv("SONAR_LORA_DISABLE_PDL", "0"))),
    # Enable CUDA compatibility mode for datacenter GPUs with older
    # driver versions than the CUDA toolkit major version of Sonar.
    "SONAR_ENABLE_CUDA_COMPATIBILITY": lambda: (
        os.environ.get("SONAR_ENABLE_CUDA_COMPATIBILITY", "0").strip().lower() in ("1", "true")
    ),
    # Path to the CUDA compatibility libraries when CUDA compatibility is enabled.
    "SONAR_CUDA_COMPATIBILITY_PATH": lambda: os.environ.get("SONAR_CUDA_COMPATIBILITY_PATH", None),
    # Skip model name validation in OpenAI API requests.
    # When set to 1, any model name will be accepted in the 'model' field
    # of API requests. This is useful for proxy/gateway scenarios where
    # the actual model is served but different names may be used in requests.
    "SONAR_SKIP_MODEL_NAME_VALIDATION": lambda: (
        os.getenv("SONAR_SKIP_MODEL_NAME_VALIDATION", "0").strip().lower() in ("1", "true")
    ),
    # Controls route logging at server startup time.
    # - "off": disable route logging
    # - "compact": group routes by API family
    # - "full": log every route individually
    "SONAR_LOG_ROUTES": env_with_choices("SONAR_LOG_ROUTES", "compact", ["off", "compact", "full"]),
    # Whether it is a scale up launch engine for elastic EP,
    # Should only be set by EngineCoreClient.
    "SONAR_ELASTIC_EP_SCALE_UP_LAUNCH": lambda: bool(int(os.getenv("SONAR_ELASTIC_EP_SCALE_UP_LAUNCH", "0"))),
    # Whether to wait for all requests to drain before sending the
    # scaling command in elastic EP.
    "SONAR_ELASTIC_EP_DRAIN_REQUESTS": lambda: bool(int(os.getenv("SONAR_ELASTIC_EP_DRAIN_REQUESTS", "0"))),
    # If set to 1, enable CUDA graph memory estimation during memory profiling.
    # This profiles CUDA graph memory usage to provide more accurate KV cache
    # memory allocation. Enabled by default as of v0.21.0
    "SONAR_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": lambda: bool(
        int(os.getenv("SONAR_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1"))
    ),
    # Force N-contiguous weight layout for all XPU unquantized linears.
    "SONAR_XPU_FORCE_N_CONTIG_WEIGHT": lambda: bool(int(os.getenv("SONAR_XPU_FORCE_N_CONTIG_WEIGHT", "0"))),
    # whether use xpu specific sample kernel
    "SONAR_XPU_USE_SAMPLER_KERNEL": lambda: bool(int(os.getenv("SONAR_XPU_USE_SAMPLER_KERNEL", "1"))),
    # Kernel backend for INC weight-only intN (WNA16) linear layers on XPU.
    # "auto" keeps the default preference order (ARK when importable, else the
    # oneDNN w4a16 path). "ark" forces the auto_round_kernel backend, "w4a16"
    # forces oneDNN int4_gemm_w4a16, and "w4a8" additionally quantizes
    # activations to per-token int8 (int4_gemm_w4a8) for large token counts.
    # The two oneDNN backends are int4-only; ARK also serves int2.
    # Which one is fastest is device-dependent, so this is left as an opt-in.
    "SONAR_XPU_INC_WNA16_BACKEND": env_with_choices(
        "SONAR_XPU_INC_WNA16_BACKEND", "auto", ["auto", "ark", "w4a16", "w4a8"]
    ),
    # Enable simple KV offload.
    "SONAR_USE_SIMPLE_KV_OFFLOAD": lambda: bool(int(os.getenv("SONAR_USE_SIMPLE_KV_OFFLOAD", "0"))),
    # Direct DCP ops default on when applicable; set to 1 to enforce or 0 to disable.
    "SONAR_USE_DIRECT_DCP_A2A": lambda: maybe_convert_bool(os.getenv("SONAR_USE_DIRECT_DCP_A2A")),
    "SONAR_USE_DIRECT_DCP_Q_GATHER": lambda: maybe_convert_bool(os.getenv("SONAR_USE_DIRECT_DCP_Q_GATHER")),
    "SONAR_USE_DIRECT_DCP_KV_GATHER": lambda: maybe_convert_bool(os.getenv("SONAR_USE_DIRECT_DCP_KV_GATHER")),
    # Whether to enable dual cuda streams for LoRA computation
    # (used by both BaseLinearLayerWithLoRA and FusedMoEWithLoRA to
    # overlap the base layer compute with the LoRA fast path).
    "SONAR_LORA_ENABLE_DUAL_STREAM": lambda: bool(int(os.getenv("SONAR_LORA_ENABLE_DUAL_STREAM", "0"))),
    # If set to 1, use Python spinloop extension to poll in a more efficient
    # way when using the mp backend.
    "SONAR_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("SONAR_USE_SPINLOOP_EXT", "0"))),
    # Comma-separated GPU_BDF=NIC_BDF pairs for RDMA NIC selection.
    # Must be set together with SONAR_NIC_SELECTION_VARS.
    "SONAR_GPU_NIC_PCIE_MAPPING": lambda: os.getenv("SONAR_GPU_NIC_PCIE_MAPPING", ""),
    # Comma-separated list of env vars to set from the GPU-NIC mapping.
    # Each entry is VAR_NAME or VAR_NAME:<suffix> (suffix appended to
    # RDMA device name). Must be set together with SONAR_GPU_NIC_PCIE_MAPPING.
    "SONAR_NIC_SELECTION_VARS": lambda: os.getenv("SONAR_NIC_SELECTION_VARS", ""),
    # If set to 1, enable the HPC fused kernels (requires the hpc package
    # (.so) and an sm100/sm103 device). Covers:
    #   * the HY V4 iHC ops -- each of the eager HYV4HCPreLayer /
    #     HYV4HCPostLayer / HYV4HCHeadLayer bodies becomes one kernel launch;
    #   * the gated-MLA output gating (attn_out * sigmoid(gate projection)),
    #     fused into gated_mla_gemm; elementwise gating only.
    # Each op additionally checks its own shape / dtype constraints and falls
    # back to the eager path when they do not hold.
    "SONAR_ENABLE_HPC_OPS": lambda: bool(int(os.getenv("SONAR_ENABLE_HPC_OPS", "0"))),
    # Whether to skip version suffix when building package
    "SONAR_SKIP_VERSION_SUFFIX": lambda: bool(int(os.getenv("SONAR_SKIP_VERSION_SUFFIX", "0"))),
    # Logging level for request JSON body.
    "SONAR_REQUEST_BODY_LOG_LEVEL": lambda: os.getenv("SONAR_REQUEST_BODY_LOG_LEVEL", "DEBUG").upper(),
}


# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    """Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_envs_cache_enabled() -> bool:
    """Checked if __getattr__ is wrapped with functools.cache."""
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """Enables caching of environment variables. This is useful for performance
    reasons, as it avoids the need to re-evaluate environment variables on
    every call.

    NOTE: Currently, it's invoked after service initialization to reduce
    runtime overhead. This also means that environment variables should NOT
    be updated after the service is initialized.
    """
    if _is_envs_cache_enabled():
        # Avoid wrapping functools.cache multiple times
        return
    # Tag __getattr__ with functools.cache
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    # Cache all environment variables
    for key in environment_variables:
        __getattr__(key)


def disable_envs_cache() -> None:
    """Resets the environment variables cache. It could be used to isolate environments
    between unit tests.
    """
    global __getattr__
    # If __getattr__ is wrapped by functions.cache, unwrap the caching layer.
    if _is_envs_cache_enabled():
        assert hasattr(__getattr__, "__wrapped__")
        __getattr__ = __getattr__.__wrapped__


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compile_factors() -> dict[str, object]:
    """Return env vars used for torch.compile cache keys.

    Start with every known Sonar env var; drop entries in `ignored_factors`;
    hash everything else. This keeps the cache key aligned across workers."""
    ignored_factors: set[str] = {
        "MAX_JOBS",
        "SONAR_PP_TRANSPORT",
        "SONAR_TRITON_JIT_WARMUP_NUM_THREADS",
        "SONAR_RPC_BASE_PATH",
        "SONAR_USE_MODELSCOPE",
        "SONAR_RINGBUFFER_WARNING_INTERVAL",
        "SONAR_DEBUG_DUMP_PATH",
        "SONAR_PORT",
        "SONAR_CACHE_ROOT",
        # Runtime memory-plan persistence; does not affect compiled graphs.
        "SONAR_ENABLE_STARTUP_PLAN",
        # Location-only derived paths: where a cache/config directory lives
        # cannot affect compiled artifacts, and hashing them means relocating
        # HOME or the XDG roots silently invalidates every compile cache
        # (SONAR_CACHE_ROOT above and
        # SONAR_FLASHINFER_AUTOTUNE_CACHE_DIR below are already ignored for
        # the same reason).
        "SONAR_XLA_CACHE_PATH",
        "SONAR_CONFIG_ROOT",
        "LD_LIBRARY_PATH",
        "SONAR_SERVER_DEV_MODE",
        "SONAR_DP_MASTER_IP",
        "SONAR_DP_MASTER_PORT",
        "SONAR_NIXL_SIDE_CHANNEL_HOST",
        "SONAR_P2P_SIDE_CHANNEL_HOST",
        "SONAR_P2P_SIDE_CHANNEL_PORT",
        "SONAR_RANDOMIZE_DP_DUMMY_INPUTS",
        "SONAR_MODEL_REDIRECT_PATH",
        "SONAR_HOST_IP",
        "SONAR_RAY_BUNDLE_INDICES",
        "SONAR_CHAT_TEMPLATE_FUEL",
        "SONAR_ELASTIC_EP_SCALE_UP_LAUNCH",
        "SONAR_FORCE_AOT_LOAD",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        # Credential; never affects compiled artifacts and must not be
        # persisted in cache_key_factors.json.
        "SONAR_API_KEY",
        "SONAR_USAGE_STATS_SERVER",
        "SONAR_NO_USAGE_STATS",
        "SONAR_DO_NOT_TRACK",
        "SONAR_LOGGING_LEVEL",
        "SONAR_LOGGING_PREFIX",
        "SONAR_LOGGING_STREAM",
        "SONAR_LOGGING_CONFIG_PATH",
        "SONAR_LOGGING_COLOR",
        "SONAR_LOGGING_VERBOSE",
        "SONAR_LOG_STATS_INTERVAL",
        "SONAR_REQUEST_LEVEL_METRICS",
        "SONAR_LOG_ROUTES",
        "SONAR_DEBUG_LOG_API_SERVER_RESPONSE",
        "SONAR_TUNED_CONFIG_FOLDER",
        "SONAR_FLASHINFER_AUTOTUNE_CACHE_DIR",
        "SONAR_FLASHINFER_AUTOTUNE_SKIP_OPS",
        "SONAR_HTTP_TIMEOUT_KEEP_ALIVE",
        "SONAR_HTTP_MAX_REQUEST_BODY_SIZE",
        "SONAR_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "SONAR_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
        "SONAR_KEEP_ALIVE_ON_ENGINE_DEATH",
        "SONAR_IMAGE_FETCH_TIMEOUT",
        "SONAR_VIDEO_FETCH_TIMEOUT",
        "SONAR_AUDIO_FETCH_TIMEOUT",
        "SONAR_MEDIA_CACHE",
        "SONAR_MEDIA_CACHE_MAX_SIZE_MB",
        "SONAR_MEDIA_CACHE_TTL_HOURS",
        "SONAR_MEDIA_FETCH_MAX_RETRIES",
        "SONAR_MEDIA_URL_ALLOW_REDIRECTS",
        "SONAR_MEDIA_LOADING_THREAD_COUNT",
        "SONAR_MAX_AUDIO_CLIP_FILESIZE_MB",
        "SONAR_MAX_AUDIO_DECODE_DURATION_S",
        "SONAR_MAX_AUDIO_DECODE_BYTES",
        "SONAR_MAX_AUDIO_PREPROCESS_WORKERS",
        "SONAR_MAX_EMBED_DECODE_BYTES",
        "SONAR_MAX_IMAGE_PIXELS",
        "SONAR_MAX_MEDIA_DOWNLOAD_SIZE_MB",
        "SONAR_MAX_MEDIA_REQUEST_DECODE_BYTES",
        "SONAR_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB",
        "SONAR_VIDEO_LOADER_BACKEND",
        "SONAR_MEDIA_CONNECTOR",
        "SONAR_OBJECT_STORAGE_SHM_BUFFER_NAME",
        "SONAR_ASSETS_CACHE",
        "SONAR_ASSETS_CACHE_MODEL_CLEAN",
        "SONAR_WORKER_MULTIPROC_METHOD",
        "SONAR_ENABLE_V1_MULTIPROCESSING",
        "SONAR_V1_OUTPUT_PROC_CHUNK_SIZE",
        "SONAR_CPU_KVCACHE_SPACE",
        "SONAR_ZENTORCH_WEIGHT_PREPACK",
        "SONAR_TEST_FORCE_LOAD_FORMAT",
        "SONAR_RELOAD_MODE",
        "SONAR_ENABLE_CUDA_COMPATIBILITY",
        "SONAR_CUDA_COMPATIBILITY_PATH",
        "SONAR_SKIP_MODEL_NAME_VALIDATION",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NO_COLOR",
        "SONAR_SKIP_VERSION_SUFFIX",
        "FORCE_COLOR",
        "SONAR_REQUEST_BODY_LOG_LEVEL",
    }

    from sonar.config.utils import normalize_value

    factors: dict[str, object] = {}
    for factor, getter in environment_variables.items():
        if factor in ignored_factors:
            continue

        try:
            raw = getter()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping environment variable %s while hashing compile factors: %s",
                factor,
                exc,
            )
            continue

        factors[factor] = normalize_value(raw)

    ray_noset_env_vars = [
        # Refer to
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/nvidia_gpu.py#L11
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/amd_gpu.py#L11
        # https://github.com/ray-project/ray/blob/b97d21dab233c2bd8ed7db749a82a1e594222b5c/python/ray/_private/accelerators/amd_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/npu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/hpu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/neuron.py#L14
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/tpu.py#L38
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/intel_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/rbln.py#L10
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
        "RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES",
    ]

    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))

    return factors
