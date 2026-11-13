# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import sonar.envs as envs
from sonar.logger import init_logger
from sonar.platforms import current_platform

logger = init_logger(__name__)

# Track whether upstream flash-attn is available on ROCm.
# Set during module initialization and never modified afterwards.
# This module-level flag avoids repeated import attempts and ensures
# consistent behavior (similar to IS_AITER_FOUND in _aiter_ops.py).
_ROCM_FLASH_ATTN_AVAILABLE = False

if current_platform.is_cuda():
    from sonar._custom_ops import reshape_and_cache_flash
    from sonar.vllm_flash_attn import (  # type: ignore[attr-defined]
        compile_flash_attn_varlen_func_from_specs,
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )

elif current_platform.is_xpu():
    from sonar import _custom_ops as ops
    from sonar._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
elif current_platform.is_rocm():
    from sonar.platforms.rocm import on_gfx1250

    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    try:
        if on_gfx1250():
            from aiter.ops.triton.mha import flash_attn_varlen_func  # type: ignore[no-redef]
        else:
            from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError:

        def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef,misc]
            package = "aiter" if on_gfx1250() else "flash-attn"
            raise ImportError(
                f"ROCm platform requires upstream {package} to be installed. Please install {package} first."
            )

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from sonar import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from sonar.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        else:
            # Fallback to FA2
            fa_version = 2

        # 2. override if passed by environment or config
        from sonar.config import get_current_sonar_config_or_none

        sonar_config = get_current_sonar_config_or_none()
        if sonar_config is not None and sonar_config.attention_config.flash_attn_version is not None:
            fa_version = sonar_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        if requires_alibi and fa_version == 3:
            logger.warning_once("Cannot use FA version 3 with ALiBi, defaulting to FA version 2.")
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once("Cannot use FA version 4 with ALiBi, defaulting to FA version 2.")
            fa_version = 2

        # Some FA3 unsupported SM90 cases can use FA4 when available.
        if fa_version == 3 and device_capability.major == 9 and is_fa_version_supported(4):
            upgrade_reason = None
            if head_size is not None and head_size > 256:
                upgrade_reason = f"FA3 does not support head_size={head_size} on SM90"
            elif has_sinks and head_size is not None and head_size_v is not None and head_size != head_size_v:
                upgrade_reason = "Diff-KV with sinks"
            elif (
                sonar_config is not None
                and sonar_config.model_config is not None
                and sonar_config.model_config.is_diffusion
            ):
                upgrade_reason = "Per-sequence causal (dynamic_causal) requires FA4"
            if upgrade_reason:
                logger.info_once(
                    "%s: upgrading FlashAttention 3 -> 4",
                    upgrade_reason,
                    scope="local",
                )
                fa_version = 4

        # FA4 currently uses batch-shape-dependent scheduling
        # heuristics on SM100+, which breaks batch invariance.
        if envs.SONAR_BATCH_INVARIANT and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, defaulting to FA version 2.",
            )
            fa_version = 2

        # FA4 head dimensions on Blackwell are limited by TMEM capacity.
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and not ((head_size == 256 and head_size_v in (None, 256)) or (head_size == 192 and head_size_v == 128))
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def is_fa_version_supported(fa_version: int) -> bool:
    try:
        from sonar.vllm_flash_attn.flash_attn_interface import (
            is_fa_version_supported as _is_fa_version_supported,
        )

        return _is_fa_version_supported(fa_version)
    except ImportError:
        return False


def flash_attn_supports_kv_cache_dtype(
    kv_cache_dtype: str = "fp8_e4m3",
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> bool:
    if kv_cache_dtype == "fp8_e5m2":
        return False
    if current_platform.is_xpu():
        return True
    fa_version = get_flash_attn_version(
        requires_alibi=requires_alibi,
        head_size=head_size,
        head_size_v=head_size_v,
        has_sinks=has_sinks,
    )
    is_sm90 = current_platform.is_device_capability_family(90)
    sm90_fp8_kv_supported = fa_version == 3 or (
        fa_version == 4 and head_size in (None, 512) and head_size_v in (None, 512)
    )
    return (sm90_fp8_kv_supported and is_sm90) or (
        fa_version == 4 and current_platform.is_device_capability_family(100)
    )


def flash_attn_supports_quant_query_input() -> bool:
    return not current_platform.is_xpu()


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    return get_flash_attn_version() in (3, 4)


def flash_attn_supports_mla():
    from sonar.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from sonar.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(3) and current_platform.is_device_capability_family(90)

            # NOTE(Lucas): FA4 CuteDSL does NOT currently support MLA's non-standard
            # head dimensions (576 for qk, 512 for v) due to TMEM capacity limits.

        except (ImportError, AssertionError):
            pass
    elif current_platform.is_xpu():
        return True
    return False


def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: sonar.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func (if available)

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.

    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
