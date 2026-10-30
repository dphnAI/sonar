# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accessors, legacy warmup and native preparation for optional ``b12x``."""

from __future__ import annotations

import importlib
import importlib.util
import weakref
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from b12x.preparation import PreparationRequest

# Logical elements and encoded bytes per block.
B12X_BLOCK_CODECS = {"iq2_xs": (256, 74), "iq2_xxs": (256, 66), "q8_0": (32, 34)}


@dataclass(frozen=True)
class B12xWarmupUnit:
    name: str
    key: Hashable
    compile: Callable[[], None]


class PreparationResourceUnavailableError(RuntimeError):
    """A native owner cannot describe or run its prepared work."""


@dataclass(frozen=True)
class B12xWorkload:
    """Exact serving specializations at the weights or cache-binding boundary."""

    stage: Literal["weights", "state"]
    token_counts: tuple[int, ...]
    fixed_token_counts: tuple[int, ...]
    output_dtype: torch.dtype
    max_tokens: int
    max_seqs: int
    lane: int = 0

    def __post_init__(self):
        if self.stage not in ("weights", "state"):
            raise ValueError("invalid native preparation stage")
        for name in ("max_tokens", "max_seqs"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.lane) is not int or self.lane < 0:
            raise ValueError("lane must be a nonnegative integer")
        counts = tuple(self.token_counts)
        if not counts or any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError("native preparation requires positive token counts")
        if counts != tuple(sorted(set(counts))) or counts[-1] > self.max_tokens:
            raise ValueError("token counts must be sorted, unique, and within capacity")
        fixed = tuple(self.fixed_token_counts)
        if fixed != tuple(sorted(set(fixed))) or not set(fixed) <= set(counts):
            raise ValueError("fixed token counts must be a sorted subset of token counts")
        if any(count >= self.max_tokens for count in fixed):
            raise ValueError("fixed token counts must be below capacity")
        object.__setattr__(self, "token_counts", counts)
        object.__setattr__(self, "fixed_token_counts", fixed)


@dataclass(frozen=True)
class B12xPreparationUnit:
    """A provider's real native requests and their execution scratch lanes."""

    name: str
    key: Hashable
    requests: tuple[PreparationRequest, ...]
    stage: Literal["weights", "state"]
    autotune: bool = True
    workspace_lanes: tuple[int, ...] = (0,)
    auxiliary_stream_requests: frozenset[str] = frozenset()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("preparation units require a family name")
        hash(self.key)
        if self.stage not in ("weights", "state"):
            raise ValueError("invalid native preparation stage")
        if type(self.autotune) is not bool:
            raise TypeError("autotune must be boolean")
        lanes = tuple(self.workspace_lanes)
        if not lanes or any(type(lane) is not int or lane < 0 for lane in lanes):
            raise ValueError("workspace lanes must be nonnegative integers")
        object.__setattr__(self, "workspace_lanes", tuple(sorted(set(lanes))))
        requests = tuple(self.requests)
        names = [request.name for request in requests]
        if len(names) != len(set(names)):
            raise ValueError("preparation unit declares duplicate request names")
        object.__setattr__(self, "requests", requests)
        auxiliary_requests = frozenset(self.auxiliary_stream_requests)
        if not auxiliary_requests <= set(names):
            raise ValueError("auxiliary stream requests must belong to the preparation unit")
        object.__setattr__(self, "auxiliary_stream_requests", auxiliary_requests)


_B12X_LAYERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_B12X_UNIT_PROVIDERS: list[weakref.ref] = []


def b12x_layer_prefix(layer: torch.nn.Module) -> str:
    """A stable, process-unique name for custom-op layer lookups."""
    prefix = getattr(layer, "prefix", None) or ""
    existing = _B12X_LAYERS.get(prefix) if prefix else None
    if not prefix or (existing is not None and existing is not layer):
        prefix = f"{prefix or type(layer).__name__}#{id(layer):x}"
    return prefix


def register_b12x_layer(name: str, layer: torch.nn.Module) -> None:
    existing = _B12X_LAYERS.get(name)
    if existing is not None and existing is not layer:
        raise ValueError(f"b12x layer name {name!r} is already registered")
    _B12X_LAYERS[name] = layer


def b12x_layer(name: str) -> torch.nn.Module:
    layer = _B12X_LAYERS.get(name)
    if layer is None:
        raise PreparationResourceUnavailableError(f"no live b12x layer named {name!r}")
    return layer


def set_b12x_preparation_provider(layer: object, provider: object) -> None:
    # A module can provide its own units; do not register it as its own child.
    object.__setattr__(layer, "b12x_preparation_provider", provider)


def register_b12x_unit_provider(provider: object, *, owner: torch.nn.Module | None = None) -> None:
    """Register a weak provider, optionally scoped to a target/draft module."""
    if owner is not None:
        object.__setattr__(provider, "b12x_preparation_owner", weakref.ref(owner))
    if not any(reference() is provider for reference in _B12X_UNIT_PROVIDERS):
        _B12X_UNIT_PROVIDERS.append(weakref.ref(provider))


def b12x_unit_providers() -> list[object]:
    live = []
    for reference in list(_B12X_UNIT_PROVIDERS):
        provider = reference()
        if provider is None:
            _B12X_UNIT_PROVIDERS.remove(reference)
        else:
            live.append(provider)
    return live


def scope_b12x_unit_calls(unit: B12xPreparationUnit, lane: int) -> B12xPreparationUnit:
    """Scope construction, execution and cleanup callbacks to the model lane."""
    from aphrodite.v1.worker.workspace import use_workspace_lane

    def wrap_factory(factory):
        def wrapped(state):
            with use_workspace_lane(lane):
                call = factory(state)

            def scoped(callback):
                if callback is None:
                    return None

                def invoke():
                    with use_workspace_lane(lane):
                        return callback()

                return invoke

            return replace(
                call,
                run=scoped(call.run),
                produce=scoped(call.produce),
                reset=scoped(call.reset),
                restore=scoped(call.restore),
                close=scoped(call.close),
            )

        return wrapped

    def wrap_calls(calls):
        if calls is None:
            return None
        if isinstance(calls, Mapping):
            return {count: wrap_factory(factory) for count, factory in calls.items()}
        return wrap_factory(calls)

    requests = tuple(
        replace(
            request,
            prepare_call=wrap_calls(request.prepare_call),
            benchmark_call=wrap_calls(request.benchmark_call),
        )
        for request in unit.requests
    )
    return replace(unit, requests=requests, workspace_lanes=(lane,))


def b12x_preparation_token_counts(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
    compile_sizes: Iterable[int] = (),
    compile_range_endpoints: Iterable[int] = (),
    speculative_tokens: int = 0,
) -> tuple[int, ...]:
    counts = {1, int(max_tokens)}
    counts.update(int(value) for value in cudagraph_capture_sizes if int(value) > 0)
    counts.update(int(value) for value in compile_sizes if int(value) > 0)
    counts.update(int(value) for value in compile_range_endpoints if int(value) > 0)
    if speculative_tokens > 0:
        counts.add(max(1, int(max_tokens) - int(speculative_tokens)))
    return tuple(sorted(counts))


_HAS_B12X = importlib.util.find_spec("b12x") is not None


def _import_submodule(module_name: str) -> ModuleType | None:
    if not _HAS_B12X:
        return None
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


_B12X_SUBMODULES = {
    module_name: _import_submodule(module_name)
    for module_name in (
        "b12x.attention.paged",
        "b12x.gemm.blockscaled",
        # TODO: Remove once B12X exposes the scale-swizzle API publicly.
        "b12x._lib.intrinsics",
        "b12x.gemm.mxfp8_linear",
        "b12x.gemm.tensor_fp8_linear",
        "b12x.moe.fused_moe",
    )
}


def has_b12x() -> bool:
    """Return whether the B12X package is installed."""
    return _HAS_B12X


def _get_submodule(module_name: str) -> ModuleType | None:
    return _B12X_SUBMODULES.get(module_name)


def get_b12x_blockscaled() -> ModuleType | None:
    return _get_submodule("b12x.gemm.blockscaled")


def get_b12x_intrinsics() -> ModuleType | None:
    return _get_submodule("b12x._lib.intrinsics")


def get_b12x_mxfp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.mxfp8_linear")


def get_b12x_tensor_fp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.tensor_fp8_linear")


def get_b12x_fused_moe() -> ModuleType | None:
    return _get_submodule("b12x.moe.fused_moe")


def get_b12x_paged_attention() -> ModuleType | None:
    return _get_submodule("b12x.attention.paged")


def b12x_warmup_token_counts(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> tuple[int, ...]:
    # B12X deduplicates shapes that select the same internal kernel policy.
    # Keep the complete serving shape set here rather than duplicating its
    # policy-selection heuristics in Aphrodite.
    counts = {1}
    counts.update(int(size) for size in cudagraph_capture_sizes if int(size) > 0)
    if int(max_tokens) > 0:
        counts.add(int(max_tokens))
    return tuple(sorted(counts))


def _same_packed_layout(current: Any, replacement: Any) -> bool:
    if type(current) is not type(replacement):
        return False
    if isinstance(current, torch.Tensor):
        return (
            current.shape == replacement.shape
            and current.stride() == replacement.stride()
            and current.dtype == replacement.dtype
            and current.device == replacement.device
        )
    if is_dataclass(current):
        return all(
            _same_packed_layout(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )
            for field in fields(current)
        )
    return bool(current == replacement)


def _copy_packed_tensors(current: Any, replacement: Any) -> None:
    if isinstance(current, torch.Tensor):
        current.copy_(replacement)
    elif is_dataclass(current):
        for field in fields(current):
            _copy_packed_tensors(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )


@torch.no_grad()
def reuse_packed_weight_storage(current: Any, replacement: Any) -> Any:
    """Reuse packed tensor addresses when a compatible weight is reloaded."""
    if current is None or not _same_packed_layout(current, replacement):
        return replacement
    _copy_packed_tensors(current, replacement)
    return current
