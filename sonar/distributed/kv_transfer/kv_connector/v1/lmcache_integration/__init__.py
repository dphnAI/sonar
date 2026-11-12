# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import importlib.util
from types import ModuleType

from .imports import enable_sonar_imports


def import_lmcache(module: str) -> ModuleType:
    enable_sonar_imports()
    return importlib.import_module(module)


def get_lmcache_adapter() -> ModuleType:
    import_lmcache("lmcache")
    # LMCache 0.5.7 moved the in-process implementation into its own package.
    prefix = "lmcache.inprocess" if importlib.util.find_spec("lmcache.inprocess") else "lmcache"
    return import_lmcache(f"{prefix}.integration.vllm.vllm_v1_adapter")
