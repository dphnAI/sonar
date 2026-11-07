# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sonar Metal runtime - high-performance LLM inference on Apple Silicon.

This runtime enables Sonar to run on Apple Silicon Macs using MLX as the
primary compute backend, with PyTorch for model loading and interoperability.
"""

import logging
import os
import sys

__version__ = "0.2.0"

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure sonar.metal logging to mirror Sonar settings."""
    from sonar.envs import SONAR_LOGGING_LEVEL

    sonar_logger = logging.getLogger("sonar")
    metal_logger = logging.getLogger("sonar.metal")
    metal_logger.setLevel(logging.getLevelName(SONAR_LOGGING_LEVEL))

    if sonar_logger.handlers and not metal_logger.handlers:
        for handler in sonar_logger.handlers:
            metal_logger.addHandler(handler)
        metal_logger.propagate = False


def _apply_macos_defaults() -> None:
    """Apply safe defaults for macOS when using the Metal plugin.

    Sonar's v1 engine launches a worker process. When the start method is
    `fork`, macOS can crash the child process if the parent has imported libraries that
    touched the Objective-C runtime (commonly surfaced as
    `objc_initializeAfterForkError`).

    Defaulting to `spawn` avoids forking a partially-initialized runtime.
    """
    if sys.platform != "darwin":
        return
    if os.environ.get("SONAR_WORKER_MULTIPROC_METHOD") is not None:
        return

    # macOS fork-safety:
    # `fork()` with an initialized Objective-C runtime is unsafe and can crash in
    # the child process (commonly observed via `objc_initializeAfterForkError`).
    # Using `spawn` starts a fresh interpreter and avoids inheriting this state.
    # See: https://www.sealiesoftware.com/blog/archive/2017/6/5/Objective-C_and_fork_in_macOS_1013.html
    os.environ["SONAR_WORKER_MULTIPROC_METHOD"] = "spawn"
    logger.debug(
        "macOS detected + Metal plugin active: defaulting SONAR_WORKER_MULTIPROC_METHOD "
        "to 'spawn' to avoid Objective-C runtime fork-safety crashes. "
        "Set SONAR_WORKER_MULTIPROC_METHOD explicitly to override."
    )


# Lazy imports to avoid loading heavy runtime dependencies on plain import.
def __getattr__(name):
    """Lazy import module components."""
    if name == "MetalConfig":
        from sonar.metal.config import MetalConfig

        return MetalConfig
    elif name == "get_config":
        from sonar.metal.config import get_config

        return get_config
    elif name == "reset_config":
        from sonar.metal.config import reset_config

        return reset_config
    elif name == "MetalPlatform":
        from sonar.metal.platform import MetalPlatform

        return MetalPlatform
    elif name == "register":
        return _register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MetalConfig",
    "MetalPlatform",
    "get_config",
    "reset_config",
    "register",
]


def _register() -> str | None:
    """Register the Metal platform with Sonar.

    Kept for compatibility with Sonar's plugin-style platform loader even
    though Metal is wired as a built-in platform.

    Returns:
        Fully qualified class name if platform is available, None otherwise
    """
    _configure_logging()
    _apply_macos_defaults()

    # Register our env vars with Sonar's registry so validate_environ()
    # does not warn about unknown SONAR_METAL_* / SONAR_MLX_* variables.
    import sonar.envs
    from sonar.metal.envs import environment_variables as metal_env_vars

    sonar.envs.environment_variables.update(metal_env_vars)

    from sonar.metal.compat import apply_compat_patches

    apply_compat_patches()

    from sonar.metal.platform import MetalPlatform

    if MetalPlatform.is_available():
        return "sonar.platforms.metal.MetalPlatform"
    return None
