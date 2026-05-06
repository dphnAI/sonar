# SPDX-License-Identifier: Apache-2.0
"""Configuration for Aphrodite Metal plugin via environment variables."""

import os
from dataclasses import dataclass
from typing import Literal

import aphrodite.metal.envs as envs

# Sentinel value indicating auto memory calculation
AUTO_MEMORY_FRACTION = -1.0

# Default memory fraction when user leaves APHRODITE_METAL_MEMORY_FRACTION as "auto"
# but enables paged attention (auto is for the MLX path).
PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION = 0.9

# Minimum blocks required for paged attention to be usable.
PAGED_ATTENTION_MIN_BLOCKS = 16

# Valid key quantization types for TurboQuant (mirrors QUANT_PARAMS in turboquant.py).
# Kept here as a plain set so config can be imported without MLX.
TURBOQUANT_VALID_K_QUANTS: frozenset[str] = frozenset(
    {"q8_0", "int8", "uint8", "q5_0", "q4_0", "int4", "uint4", "int2", "uint2"}
)

# Valid value quantization types for TurboQuant.
# V uses Lloyd-Max quantization with FWHT rotation.
TURBOQUANT_VALID_V_QUANTS: frozenset[str] = frozenset(
    {"q2_0", "q3_0", "q4_0", "q5_0", "q8_0"}
)

MultimodalMode = Literal["auto", "text-only-compat", "multimodal-native"]
VALID_MULTIMODAL_MODES: frozenset[MultimodalMode] = frozenset(
    {"auto", "text-only-compat", "multimodal-native"}
)


@dataclass
class MetalConfig:
    """Configuration for Aphrodite Metal plugin."""

    memory_fraction: float  # -1.0 means "auto" (calculate minimal needed)
    use_mlx: bool
    mlx_device: Literal["gpu", "cpu"]
    debug: bool
    use_paged_attention: bool = True
    kv_sharing_fast_prefill: bool = False
    multimodal_mode: MultimodalMode = "auto"
    turboquant: bool = False  # Enable TurboQuant KV cache compression
    k_quant: str = "q8_0"  # Key quantization type: q8_0, q4_0, int8, uint8, etc.
    v_quant: str = "q3_0"  # Value quantization type: q2_0, q3_0, q4_0, q5_0 (Lloyd-Max)

    def __post_init__(self) -> None:
        if not self.use_paged_attention and not self.is_auto_memory:
            raise ValueError(
                f"APHRODITE_METAL_MEMORY_FRACTION={self.memory_fraction} is only "
                "supported with paged attention (the default). "
                "The MLX KV cache path (APHRODITE_METAL_USE_PAGED_ATTENTION=0) "
                "requires APHRODITE_METAL_MEMORY_FRACTION=auto."
            )

        if self.kv_sharing_fast_prefill and not self.use_paged_attention:
            raise ValueError(
                "APHRODITE_METAL_KV_SHARING_FAST_PREFILL requires paged attention. "
                "Enable APHRODITE_METAL_USE_PAGED_ATTENTION=1 or set "
                "APHRODITE_METAL_KV_SHARING_FAST_PREFILL=0."
            )

        if self.use_paged_attention and not self.is_auto_memory:
            if not (0 < self.memory_fraction <= 1):
                raise ValueError(
                    f"Invalid APHRODITE_METAL_MEMORY_FRACTION={self.memory_fraction}. "
                    "Must be a finite value in (0, 1] when paged attention is enabled."
                )

        if self.multimodal_mode not in VALID_MULTIMODAL_MODES:
            available = ", ".join(sorted(VALID_MULTIMODAL_MODES))
            raise ValueError(
                f"Invalid APHRODITE_METAL_MULTIMODAL_MODE={self.multimodal_mode!r}. "
                f"Available modes: {available}."
            )

        self._validate_turboquant()

    def _validate_turboquant(self) -> None:
        """Validate TurboQuant configuration."""
        if self.turboquant:
            if not self.use_paged_attention:
                raise ValueError(
                    "turboquant requires paged attention. "
                    "TurboQuant KV cache compression only works with paged attention."
                )
            if self.k_quant not in TURBOQUANT_VALID_K_QUANTS:
                available = ", ".join(sorted(TURBOQUANT_VALID_K_QUANTS))
                raise ValueError(
                    f"Invalid k_quant={self.k_quant!r}. "
                    f"Available quantization types: {available}"
                )
            if self.v_quant not in TURBOQUANT_VALID_V_QUANTS:
                available = ", ".join(sorted(TURBOQUANT_VALID_V_QUANTS))
                raise ValueError(
                    f"Invalid v_quant={self.v_quant!r}. "
                    f"Available quantization types: {available}"
                )

    @property
    def is_auto_memory(self) -> bool:
        """Check if memory fraction is set to auto mode."""
        return self.memory_fraction == AUTO_MEMORY_FRACTION

    @classmethod
    def from_env(cls) -> "MetalConfig":
        """Load configuration from environment variables."""
        memory_fraction_str = envs.APHRODITE_METAL_MEMORY_FRACTION
        if memory_fraction_str.lower() == "auto":
            memory_fraction = AUTO_MEMORY_FRACTION
        else:
            try:
                memory_fraction = float(memory_fraction_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid APHRODITE_METAL_MEMORY_FRACTION={memory_fraction_str!r}. "
                    "Must be 'auto' or a numeric value in (0, 1]."
                ) from e

        use_paged_attention = envs.APHRODITE_METAL_USE_PAGED_ATTENTION
        kv_sharing_fast_prefill = envs.APHRODITE_METAL_KV_SHARING_FAST_PREFILL
        if (
            not use_paged_attention
            and "APHRODITE_METAL_KV_SHARING_FAST_PREFILL" not in os.environ
        ):
            kv_sharing_fast_prefill = False

        # TurboQuant config is set via --additional-config, not env vars.
        # See MetalPlatform.check_and_update_config() for how it's applied.
        return cls(
            memory_fraction=memory_fraction,
            use_mlx=envs.APHRODITE_METAL_USE_MLX,
            mlx_device=envs.APHRODITE_MLX_DEVICE,  # type: ignore[arg-type]
            debug=envs.APHRODITE_METAL_DEBUG,
            use_paged_attention=use_paged_attention,
            kv_sharing_fast_prefill=kv_sharing_fast_prefill,
            multimodal_mode=envs.APHRODITE_METAL_MULTIMODAL_MODE,  # type: ignore[arg-type]
        )


# Global config instance
_config: MetalConfig | None = None


def get_config() -> MetalConfig:
    """Get the global Metal configuration."""
    global _config
    if _config is None:
        _config = MetalConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None


def should_use_contiguous_kv_fast_path(
    config: MetalConfig,
    *,
    model_config: object | None,
    scheduler_config: object,
) -> bool:
    """Return whether Metal should prefer MLX's contiguous KV cache.

    Paged attention is still the default for higher concurrency and features
    that need block-managed KV state.  For dense, low-concurrency text serving,
    MLX's contiguous cache is currently much faster on decode and does not
    require an environment variable from the user.
    """
    return (
        "APHRODITE_METAL_USE_PAGED_ATTENTION" not in os.environ
        and config.use_paged_attention
        and config.is_auto_memory
        and not config.turboquant
        and model_config is not None
        and not getattr(model_config, "is_hybrid", False)
        and getattr(scheduler_config, "max_num_seqs") <= 2
    )


def enable_contiguous_kv_fast_path(config: MetalConfig) -> None:
    """Switch a Metal config to the contiguous MLX KV cache path."""
    config.use_paged_attention = False
    config.kv_sharing_fast_prefill = False
