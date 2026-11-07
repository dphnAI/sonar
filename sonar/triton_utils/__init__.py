# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from typing import TYPE_CHECKING, Any

import sonar.envs as envs
from sonar.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

gluon: Any
gl: Any
aggregate: Any

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    gluon = import_module("triton.experimental.gluon")
    gl = import_module("triton.experimental.gluon.language")
    aggregate = import_module("triton.language.core")._aggregate
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    gluon = TritonLanguagePlaceholder()
    gl = TritonLanguagePlaceholder()
    aggregate = TritonLanguagePlaceholder()

from sonar.triton_utils.tensor_descriptor import use_tensor_descriptor  # noqa: E402

# Replace Autotuner.run so it always picks the first valid configuration, to
# eliminate autotune variability. Installed here rather than in
# sonar.env_override because importing this package resolves current_platform,
# which must not happen during `import sonar`.
if envs.SONAR_TRITON_FORCE_FIRST_CONFIG:
    from sonar.triton_utils.force_first_config import (
        install as _install_force_first_config,
    )

    _install_force_first_config()

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453


def get_cache_invalidating_env_vars() -> dict[str, str]:
    """Read Triton's cache environment without eagerly importing its compiler."""
    get_env_vars = import_module("triton._C.libtriton").get_cache_invalidating_env_vars

    return get_env_vars()


__all__ = [
    "HAS_TRITON",
    "get_cache_invalidating_env_vars",
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "gluon",
    "gl",
    "aggregate",
    "use_tensor_descriptor",
]
