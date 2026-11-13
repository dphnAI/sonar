# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import torch

from sonar.config.quantization import (
    _ONLINE_SHORTHANDS,
    QuantizationConfigArgs,
    QuantSpec,
)
from sonar.logger import init_logger
from sonar.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from sonar.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from sonar.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sonar.model_executor.layers.quantization import QuantizationMethods
from sonar.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sonar.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sonar.model_executor.layers.quantization.input_hadamard import (
    InputHadamardLinearMethod,
)
from sonar.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
    Fp8PtpcOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
    OnlineLinearBase,
)
from sonar.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)
from sonar.model_executor.layers.quantization.online.lut_b import (
    LutBOnlineMoEMethod,
)
from sonar.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase
from sonar.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from sonar.model_executor.layers.quantization.online.mxfp6 import (
    Mxfp6OnlineLinearMethod,
    Mxfp6OnlineMoEMethod,
)
from sonar.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
    Mxfp8OnlineMoEMethod,
)
from sonar.model_executor.layers.quantization.online.nf12 import (
    Nf12OnlineLinearMethod,
)
from sonar.model_executor.layers.quantization.online.nf12_moe import (
    Nf12OnlineMoEMethod,
)
from sonar.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
)
from sonar.model_executor.layers.quantization.utils.config_utils import (
    find_matching_patterns,
    get_layer_name_after_index,
)
from sonar.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from sonar.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from sonar.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static32,
    kInt8StaticChannelSym,
    kLutBStatic,
    kMxfp4Static,
    kMxfp6E2M3Static,
    kMxfp6E3M2Static,
    kMxfp8Static,
    kNf12,
    kNvfp4Static,
)

logger = init_logger(__name__)


class OnlineQuantizationSource(StrEnum):
    """Supported online quantization configuration sources."""

    linear = "linear"  # LinearBase
    moe = "moe"  # RoutedExperts
    targets = "targets"


# Online dispatch tables, keyed by the QuantSpec.weight QuantKey.
_ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineLinearMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineLinearMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineLinearMethod,
    kMxfp8Static: Mxfp8OnlineLinearMethod,
    kMxfp6E2M3Static: Mxfp6OnlineLinearMethod,
    kMxfp6E3M2Static: Mxfp6OnlineLinearMethod,
    kMxfp4Static: Mxfp4OnlineLinearMethod,
    kNf12: Nf12OnlineLinearMethod,
}

# Online linear methods validated with `QuantSpec.hadamard_size`.
_ONLINE_HADAMARD_LINEAR_METHODS: frozenset[type] = frozenset({Mxfp4OnlineLinearMethod})

_ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineMoEMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineMoEMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineMoEMethod,
    kMxfp8Static: Mxfp8OnlineMoEMethod,
    kMxfp4Static: Mxfp4OnlineMoEMethod,
    kInt8StaticChannelSym: Int8OnlineMoEMethod,
    kNvfp4Static: Nvfp4OnlineMoEMethod,
    kMxfp6E2M3Static: Mxfp6OnlineMoEMethod,
    kMxfp6E3M2Static: Mxfp6OnlineMoEMethod,
    kNf12: Nf12OnlineMoEMethod,
    kLutBStatic: LutBOnlineMoEMethod,
}

# Quantizers for full-precision fused shared-expert weights, keyed by the routed
# experts' weight QuantKey. Each shard is quantized to the checkpoint layout on
# arrival, so only block-local schemes are supported.
ONLINE_SHARED_EXPERT_QUANTIZERS: dict[QuantKey, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {
    kMxfp4Static: mxfp4_quantize,
}

# Dequantizers `(weight, weight_scale) -> high precision weight` of quantized fused
# shared-expert checkpoint weights, keyed by the checkpoint weight QuantKey.
ONLINE_SHARED_EXPERT_DEQUANTIZERS: dict[QuantKey, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    kMxfp8Static: dequant_mxfp8_to_bf16,
}


def _resolved_target(target: str | QuantSpec) -> tuple:
    """Comparable identity of a `quantization_config.targets` value.

    Returns one `(weight, activation, activation was explicit, hadamard_size)`
    tuple per layer kind (linear, MoE), or `None` for a kind the target does not
    define: a shorthand name and the equivalent `QuantSpec` therefore compare
    equal, while an omitted activation still differs from an explicit `null`.
    """
    if isinstance(target, str):
        shorthand = _ONLINE_SHORTHANDS[target]
        # A one-sided shorthand applies its spec to either layer kind, like the
        # equivalent `QuantSpec` does, so compare it that way.
        specs = (
            shorthand.linear or shorthand.moe,
            shorthand.moe or shorthand.linear,
        )
    else:
        specs = (target, target)
    return tuple(
        None
        if spec is None
        else (
            spec.weight,
            spec.activation,
            "activation" in spec.fields_set,
            spec.hadamard_size,
        )
        for spec in specs
    )


def _find_matching_targets(
    prefix: str,
    targets: Mapping[str, str | QuantSpec],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> list[str]:
    per_shard_matches = find_matching_patterns(prefix, targets, fused_mapping, use_fnmatch=True)
    if all(len(matches) == 0 for matches in per_shard_matches):
        return []
    if any(len(matches) == 0 for matches in per_shard_matches):
        raise ValueError(
            f"Found unmatched shards for {prefix}: {per_shard_matches}. Sonar "
            "requires all shards of a fused layer to match a target."
        )
    if any(len(matches) > 1 for matches in per_shard_matches):
        raise ValueError(
            f"Found multiple quantization_config.targets matches for the "
            f"shards of {prefix}: {per_shard_matches}. Each shard may match "
            "at most one target."
        )

    matched_patterns = [next(iter(matches)) for matches in per_shard_matches]
    target_values = [_resolved_target(targets[pattern]) for pattern in matched_patterns]
    if any(target != target_values[0] for target in target_values[1:]):
        raise ValueError(
            f"Found different quantization_config.targets values for the "
            f"shards of {prefix}: {matched_patterns}. Sonar requires all "
            "shards of a fused layer to use the same target."
        )
    return [matched_patterns[0]]


class OnlineQuantizationConfig(QuantizationConfig):
    """Model-level config for online quantization (quantize fp16/bf16 weights
    during model loading, without requiring a pre-quantized checkpoint)."""

    def __init__(
        self,
        args: QuantizationConfigArgs,
    ) -> None:
        super().__init__()
        if args.linear is None and args.moe is None and args.targets is None:
            raise ValueError(
                "OnlineQuantizationConfig requires at least one of "
                "quantization_config.linear, quantization_config.moe, or "
                "quantization_config.targets to be set."
            )
        self.args = args
        self.ignored_layers: list[str] = args.ignore
        self.quantized_layers: dict[str, tuple[str, str, str | None]] = {}
        self.online_quantization_time = 0.0

    @property
    def quantized_layer_summaries(self) -> list[str]:
        counts: dict[tuple[str, str, str | None, str], int] = {}
        for layer_name, (
            source,
            quant_key_str,
            target_pattern,
        ) in self.quantized_layers.items():
            key = (
                get_layer_name_after_index(layer_name),
                source,
                target_pattern,
                quant_key_str,
            )
            counts[key] = counts.get(key, 0) + 1

        summaries = []
        # Build summary entries as
        # `self_attn.o_proj: 24 (from targets: re:.*self_attn\.o_proj, mxfp4`
        for (layer_type, source, target_pattern, quant_key_str), count in sorted(counts.items()):
            pattern_prefix = f"{target_pattern}, " if target_pattern else ""
            summaries.append(f"{layer_type}: {count} (from {source}: {pattern_prefix}{quant_key_str})")
        return summaries

    def _resolve_spec(self, prefix: str, base: QuantSpec | None) -> QuantSpec | None:
        """Resolve ordered precision rules for a module.

        `ignore` remains an unconditional compatibility escape hatch. The
        newer overrides are evaluated in declaration order, so a later rule
        can refine (or undo) an earlier broad rule.
        """
        if should_ignore_layer(
            prefix,
            ignore=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return None

        spec = base
        for override in self.args.overrides:
            if not should_ignore_layer(
                prefix,
                ignore=[override.pattern],
                fused_mapping=self.packed_modules_mapping,
            ):
                continue
            if override.weight == "bf16":
                spec = None
                continue

            fields = {name: getattr(spec, name) for name in spec.fields_set} if spec is not None else {}
            if override.weight is not None:
                fields["weight"] = override.weight
            if override.activation is not None:
                fields["activation"] = override.activation
            spec = QuantSpec(**fields)
        return spec

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "online"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Note: as more online quant schemes will be added, this
        # value will become the minimum across all supported schemes.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineQuantizationConfig":
        raise NotImplementedError(
            "OnlineQuantizationConfig does not support loading from a "
            "checkpoint config. Use quantization_config or "
            "quantization='fp8_per_tensor'/'fp8_per_block' instead."
        )

    def _get_method_cls(
        self,
        spec: QuantSpec | None,
        table: dict[QuantKey, type],
        layer_cls: type[torch.nn.Module],
    ) -> type | None:
        """Resolve the online method class for a layer's quantization spec.

        Args:
            spec: Quantization specification to resolve.
            table: Mapping from weight quantization keys to method classes.
            layer_cls: Type of the layer that will use the resolved method.

        Returns:
            The matching method class, or None when `spec` has no weight
            quantization.

        """
        if spec is None or spec.weight is None:
            return None
        # Load-time gfx942 requant, not online conversion. Mxfp4MoEMethod owns it.
        if spec.weight == kInt4Static32:
            return None
        cls = table.get(spec.weight)
        if cls is None:
            raise ValueError(
                f"online quantization for {layer_cls.__name__} with "
                f"weight={spec.weight} is not supported; supported weight "
                f"keys: {sorted(str(k) for k in table)}"
            )
        # Only the validated linear methods can rotate weights and activations.
        if spec.hadamard_size is not None and cls not in _ONLINE_HADAMARD_LINEAR_METHODS:
            raise NotImplementedError(
                f"hadamard_size is not yet supported for online {cls.__name__}; "
                "supported: "
                f"{sorted(c.__name__ for c in _ONLINE_HADAMARD_LINEAR_METHODS)}"
            )
        if cls in (Mxfp6OnlineLinearMethod, Mxfp6OnlineMoEMethod):
            from sonar.model_executor.kernels.linear.mxfp6 import CutedslMxfp6LinearKernel

            supported, reason = CutedslMxfp6LinearKernel.is_supported()
            if not supported:
                logger.warning_once("Keeping MXFP6-targeted layers in BF16 because %s", reason)
                return None
        return cls

    def resolve_quant_method_cls(
        self,
        layer_cls: type[torch.nn.Module],
        prefix: str,
        *,
        layer: torch.nn.Module | None = None,
    ) -> tuple[OnlineQuantizationSource, str, str | None, QuantSpec, type] | None:
        """Resolve quantization metadata and method class without instantiating it.

        Args:
            layer_cls: Type of layer for which to resolve online quantization.
            prefix: Fully qualified layer name.
            layer: Optional instance for runtime capability checks. Class-only
                callers resolve configured metadata without these checks.

        Returns:
            A tuple of source, quantization key string, target pattern, spec
            with its resolved activation key, and method class. Returns None
            when online quantization does not apply to the layer. The returned
            spec's `fields_set` is not user input.

        """
        quant_spec: QuantSpec | None
        if self.args.targets is not None:
            resolved_pattern = self._resolve_targets_quant_method_metadata(prefix, layer_cls)
            if resolved_pattern is None:
                return None
            source, quant_key_str, target_pattern, quant_spec, table = resolved_pattern
        else:
            if issubclass(layer_cls, LinearBase):
                source = OnlineQuantizationSource.linear
                quant_spec = self.args.linear
                table = _ONLINE_LINEAR_METHODS
            elif issubclass(layer_cls, RoutedExperts):
                source = OnlineQuantizationSource.moe
                quant_spec = self.args.moe
                table = _ONLINE_MOE_METHODS
            else:
                return None

            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                use_fnmatch=True,
            ):
                return None
            quant_key_str = str(quant_spec)
            target_pattern = None

        resolved_spec = self._resolve_spec(prefix, quant_spec)
        if resolved_spec is not quant_spec:
            quant_key_str = str(resolved_spec)
        quant_spec = resolved_spec
        quant_method_cls = self._get_method_cls(quant_spec, table, layer_cls)
        if quant_method_cls is None:
            return None
        assert quant_spec is not None
        if quant_method_cls is Nf12OnlineMoEMethod and layer is not None:
            supported, reason = quant_method_cls.can_implement(layer)
            if not supported:
                logger.warning_once("Skipping NF12 for unsupported experts: %s", reason)
                return None
        assert issubclass(quant_method_cls, (OnlineMoEMethodBase, OnlineLinearBase))
        explicit = "activation" in quant_spec.fields_set
        quant_spec = QuantSpec(
            weight=quant_spec.weight,
            activation=quant_method_cls.resolve_activation_quant_key(
                quant_spec.activation if explicit else getattr(quant_method_cls, "activation_quant_key", None),
                explicit=explicit,
            ),
            # Carried so `get_quant_method` can wrap the method in
            # `InputHadamardLinearMethod`.
            hadamard_size=quant_spec.hadamard_size,
        )
        return source, quant_key_str, target_pattern, quant_spec, quant_method_cls

    def _resolve_targets_quant_method_metadata(
        self, prefix: str, layer_cls: type[torch.nn.Module]
    ) -> tuple[OnlineQuantizationSource, str, str, QuantSpec, dict[QuantKey, type]] | None:
        """Resolve target-pattern quantization metadata and dispatch table.

        Args:
            prefix: Fully qualified layer name.
            layer_cls: Type of layer matched against configured target patterns.

        Returns:
            A tuple of source, quantization key string, target pattern, spec,
            and dispatch table. Returns None when no pattern applies or the
            layer is ignored.

        """
        assert self.args.targets is not None
        ignored = should_ignore_layer(
            prefix,
            ignore=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            use_fnmatch=True,
        )
        matches = _find_matching_targets(prefix, self.args.targets, fused_mapping=self.packed_modules_mapping)
        if ignored and matches:
            raise ValueError(
                f"Layer {prefix} matches both quantization_config.ignore "
                f"and quantization_config.targets ({matches}); a layer may "
                f"not be referenced by both."
            )
        if ignored or not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"Layer {prefix} matches multiple quantization_config."
                f"targets patterns: {matches}. Each layer may match at most "
                f"one target."
            )
        target_pattern = matches[0]
        target = self.args.targets[target_pattern]
        if isinstance(target, str):
            shorthand = _ONLINE_SHORTHANDS[target]
            quant_key_str = target
            quant_spec = shorthand.linear if issubclass(layer_cls, LinearBase) else shorthand.moe
        else:
            quant_spec = target
            quant_key_str = str(quant_spec)
        if issubclass(layer_cls, LinearBase):
            table = _ONLINE_LINEAR_METHODS
        elif issubclass(layer_cls, RoutedExperts):
            table = _ONLINE_MOE_METHODS
        else:
            raise ValueError(
                f"Layer {prefix} was matched by quantization_config.targets "
                f"({target_pattern}), but online quantization is not supported for "
                f"{layer_cls.__name__}."
            )
        if quant_spec is None:
            raise ValueError(
                f"targets pattern {target_pattern} = {target} does "
                f"not define a QuantSpec for {layer_cls.__name__} layers "
                f"(matched at {prefix})."
            )
        return (
            OnlineQuantizationSource.targets,
            quant_key_str,
            target_pattern,
            quant_spec,
            table,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> "QuantizeMethodBase | None":
        # `targets` takes precedence over `moe` and `linear` and is exclusive.
        resolved = self.resolve_quant_method_cls(type(layer), prefix, layer=layer)
        if resolved is not None:
            source, quant_key_str, target_pattern, quant_spec, quant_method_cls = resolved
            if quant_method_cls is Mxfp6OnlineMoEMethod:
                supported, reason = Mxfp6OnlineMoEMethod.can_implement(layer.moe_config)
                if not supported:
                    logger.warning_once("Keeping MXFP6-targeted MoE layers in BF16 because %s", reason)
                    return UnquantizedFusedMoEMethod(layer.moe_config)
            self.quantized_layers[prefix] = (
                source.value,
                quant_key_str,
                target_pattern,
            )
            if quant_method_cls is Mxfp6OnlineLinearMethod:
                return quant_method_cls(quant_spec)
            if quant_method_cls is Mxfp6OnlineMoEMethod:
                return quant_method_cls(layer=layer, spec=quant_spec)
            assert issubclass(quant_method_cls, (OnlineMoEMethodBase, OnlineLinearBase))
            kwargs = {}
            if quant_method_cls.accepts_activation_override:
                kwargs["activation_quant_key"] = quant_spec.activation
            if isinstance(layer, RoutedExperts):
                assert issubclass(quant_method_cls, OnlineMoEMethodBase)
                return quant_method_cls(moe=layer.moe_config, **kwargs)

            assert issubclass(quant_method_cls, OnlineLinearBase)
            if quant_spec.hadamard_size is not None:
                return InputHadamardLinearMethod(
                    quant_method_cls(**kwargs),
                    quant_spec.hadamard_size,
                    rotate_weight=True,
                )
            return quant_method_cls(**kwargs)

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        return None
