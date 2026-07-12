# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy facade for the optional ``humming`` package.

Aphrodite code should import humming symbols from here so that ``import humming``
(which has import-time side effects) is deferred until first use. Add new
symbols by appending one entry to ``_EXPORTS`` as ``"module.path:attr"``,
or ``"module.path"`` for a whole-module re-export.
"""

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "dtypes": "humming.dtypes",
    "DataType": "humming.dtypes:DataType",
    "GemmType": "humming.config:GemmType",
    "WeightScaleType": "humming.config:WeightScaleType",
    "HummingMethod": "humming.layer:HummingMethod",
    "HummingLayerMeta": "humming.layer:HummingLayerMeta",
    "BaseInputSchema": "humming.schema:BaseInputSchema",
    "BaseWeightSchema": "humming.schema:BaseWeightSchema",
    "HummingInputSchema": "humming.schema:HummingInputSchema",
    "HummingWeightSchema": "humming.schema:HummingWeightSchema",
    "quantize_weight": "humming.utils.weight:quantize_weight",
    "AWQWeightSchema": "humming.schema:AWQWeightSchema",
    "BitnetWeightSchema": "humming.schema:BitnetWeightSchema",
    "ModeloptMxfp8WeightSchema": "humming.schema.modelopt:ModeloptMxfp8WeightSchema",
    "ModeloptNvfp4InputSchema": "humming.schema.modelopt:ModeloptNvfp4InputSchema",
    "ModeloptNvfp4WeightSchema": "humming.schema.modelopt:ModeloptNvfp4WeightSchema",
    "CompressedTensorsInputSchema": "humming.schema:CompressedTensorsInputSchema",
    "CompressedTensorsWeightSchema": "humming.schema:CompressedTensorsWeightSchema",
    "Fp8InputSchema": "humming.schema:Fp8InputSchema",
    "Fp8WeightSchema": "humming.schema.fp8:Fp8WeightSchema",
    "Mxfp4WeightSchema": "humming.schema:Mxfp4WeightSchema",
    "GptOssMxfp4WeightSchema": "humming.schema:GptOssMxfp4WeightSchema",
    "GPTQWeightSchema": "humming.schema:GPTQWeightSchema",
}


_sm110_compat_applied = False


def _apply_sm110_compat() -> None:
    """Humming ships no sm110 support: its heuristics map has no 11.0 entry
    and Jetson does not expose the NVML clock/bus-width queries its roofline
    estimates rely on. The sm100 heuristics work unchanged on sm110, so map
    them in and pin the two roofline numbers to Thor's."""
    global _sm110_compat_applied
    if _sm110_compat_applied:
        return
    _sm110_compat_applied = True
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        return
    from humming.tune import heuristics_map
    from humming.tune.sm100 import Sm100Heuristics
    from humming.utils import device as humming_device

    heuristics_map.setdefault(110, Sm100Heuristics)
    humming_device.calculate_gpu_bandwidth = lambda gpu_index=0: 273.0
    humming_device.estimate_tensorcore_max_tops = lambda gpu_index=0: 250


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'aphrodite.utils.humming' has no attribute {name!r}")
    _apply_sm110_compat()
    if ":" in spec:
        mod_path, attr = spec.split(":", 1)
        obj = getattr(importlib.import_module(mod_path), attr)
    else:
        obj = importlib.import_module(spec)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
