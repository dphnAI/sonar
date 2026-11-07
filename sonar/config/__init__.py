# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from sonar.config.attention import AttentionConfig, HiSparseConfig
from sonar.config.aux_output import AuxOutputConfig
from sonar.config.cache import CacheConfig
from sonar.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    PassConfig,
)
from sonar.config.device import DeviceConfig
from sonar.config.diffusion import DiffusionConfig
from sonar.config.dphn import DphnNetworkConfig
from sonar.config.ec_manager_config import EncoderCacheManagerConfig
from sonar.config.ec_transfer import ECTransferConfig
from sonar.config.engram import EngramConfig
from sonar.config.fault_tolerance import FaultToleranceConfig
from sonar.config.kernel import KernelConfig
from sonar.config.kv_events import KVEventsConfig
from sonar.config.kv_transfer import KVTransferConfig
from sonar.config.load import LoadConfig
from sonar.config.logging import LoggingConfig
from sonar.config.lora import LoRAConfig
from sonar.config.mamba import MambaConfig
from sonar.config.model import (
    ModelConfig,
    iter_architecture_defaults,
    str_dtype_to_torch_dtype,
    try_match_architecture_defaults,
)
from sonar.config.multimodal import MultiModalConfig
from sonar.config.observability import ObservabilityConfig
from sonar.config.offload import (
    OffloadBackend,
    OffloadConfig,
    PrefetchOffloadConfig,
    UVAOffloadConfig,
)
from sonar.config.parallel import EPLBConfig, ParallelConfig
from sonar.config.pooler import PoolerConfig
from sonar.config.profiler import ProfilerConfig
from sonar.config.reasoning import ReasoningConfig
from sonar.config.scheduler import SchedulerConfig
from sonar.config.sonar import (
    SonarConfig,
    get_cached_compilation_config,
    get_current_sonar_config,
    get_current_sonar_config_or_none,
    get_layers_from_sonar_config,
    set_current_sonar_config,
)
from sonar.config.speculative import SpeculativeConfig
from sonar.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from sonar.config.structured_outputs import StructuredOutputsConfig
from sonar.config.utils import (
    ConfigType,
    SupportsMetricsInfo,
    config,
    get_attr_docs,
    is_init_field,
    replace,
    update_config,
)
from sonar.config.watchdog_config import WatchdogConfig
from sonar.config.watermarking import WatermarkConfig
from sonar.config.weight_transfer import WeightTransferConfig

# __all__ should only contain classes and functions.
# Types and globals should be imported from their respective modules.
__all__ = [
    # From sonar.config.aux_output
    "AuxOutputConfig",
    # From sonar.config.attention
    "AttentionConfig",
    "HiSparseConfig",
    # From sonar.config.cache
    "CacheConfig",
    # From sonar.config.compilation
    "CompilationConfig",
    "CompilationMode",
    "CUDAGraphMode",
    "PassConfig",
    # From sonar.config.device
    "DeviceConfig",
    # From sonar.config.diffusion
    "DiffusionConfig",
    # From sonar.config.dphn
    "DphnNetworkConfig",
    # From sonar.config.ec_manager_config
    "EncoderCacheManagerConfig",
    # From sonar.config.ec_transfer
    "ECTransferConfig",
    # From sonar.config.engram
    "EngramConfig",
    # From sonar.config.kernel
    "KernelConfig",
    # From sonar.config.kv_events
    "KVEventsConfig",
    # From sonar.config.kv_transfer
    "KVTransferConfig",
    # From sonar.config.load
    "LoadConfig",
    # From sonar.config.logging
    "LoggingConfig",
    # From sonar.config.lora
    "LoRAConfig",
    # From sonar.config.mamba
    "MambaConfig",
    # From sonar.config.model
    "ModelConfig",
    "iter_architecture_defaults",
    "str_dtype_to_torch_dtype",
    "try_match_architecture_defaults",
    # From sonar.config.multimodal
    "MultiModalConfig",
    # From sonar.config.observability
    "ObservabilityConfig",
    # From sonar.config.offload
    "OffloadBackend",
    "OffloadConfig",
    "PrefetchOffloadConfig",
    "UVAOffloadConfig",
    # From sonar.config.parallel
    "EPLBConfig",
    "ParallelConfig",
    # From sonar.config.pooler
    "PoolerConfig",
    # From sonar.config.reasoning
    "ReasoningConfig",
    # From sonar.config.scheduler
    "SchedulerConfig",
    # From sonar.config.speculative
    "SpeculativeConfig",
    # From sonar.config.speech_to_text
    "SpeechToTextConfig",
    "SpeechToTextParams",
    # From sonar.config.structured_outputs
    "StructuredOutputsConfig",
    # From sonar.config.profiler
    "ProfilerConfig",
    # From sonar.config.fault_tolerance
    "FaultToleranceConfig",
    # From sonar.config.utils
    "ConfigType",
    "SupportsMetricsInfo",
    "config",
    "get_attr_docs",
    "is_init_field",
    "replace",
    "update_config",
    # From sonar.config.sonar
    "SonarConfig",
    "get_cached_compilation_config",
    "get_current_sonar_config",
    "get_current_sonar_config_or_none",
    "set_current_sonar_config",
    "get_layers_from_sonar_config",
    "WatchdogConfig",
    "WeightTransferConfig",
    "WatermarkConfig",
]
