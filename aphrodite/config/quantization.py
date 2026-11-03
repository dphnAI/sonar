# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code=call-arg

from typing import TYPE_CHECKING, Annotated, Any, Literal

import regex as re
from pydantic import (
    Field,
    GetPydanticSchema,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from aphrodite.config.utils import config
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static32,
    kInt8Dynamic32Sym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
    kNvfp4DynamicToken,
    kNvfp4Static,
)

# User-facing names addressable from quantization_config.
QUANT_KEY_NAMES: dict[str, QuantKey] = {
    "fp8_per_tensor_static": kFp8StaticTensorSym,
    "fp8_per_tensor_dynamic": kFp8DynamicTensorSym,
    "fp8_per_token": kFp8DynamicTokenSym,
    "fp8_per_channel_static": kFp8StaticChannelSym,
    "fp8_per_block_static": kFp8Static128BlockSym,
    "fp8_per_block_dynamic": kFp8Dynamic128Sym,
    "mxfp8": kMxfp8Dynamic,
    "mxfp4": kMxfp4Dynamic,
    "mxfp6_e2m3": kMxfp6E2M3Static,
    "mxfp6_e2m3_dynamic": kMxfp6E2M3Dynamic,
    "mxfp6_e3m2": kMxfp6E3M2Static,
    "mxfp6_e3m2_dynamic": kMxfp6E3M2Dynamic,
    "nvfp4_per_token": kNvfp4DynamicToken,
    "int8_per_channel_static": kInt8StaticChannelSym,
    # Dynamic per-32-element int8 activations (RDNA3 MXFP4 W4A8 GEMV).
    "int8_per_group_32": kInt8Dynamic32Sym,
    # Load-time MXFP4-to-int4 for Kimi-K3 on gfx942.
    "int4_per_group_32": kInt4Static32,
}


def _coerce_quant_key(v: Any) -> QuantKey | None:
    if v is None or isinstance(v, QuantKey):
        return v
    if not isinstance(v, str):
        raise TypeError(f"expected str or QuantKey, got {type(v).__name__}")
    try:
        return QUANT_KEY_NAMES[v]
    except KeyError:
        raise ValueError(f"unknown quantization name {v!r}; expected one of {sorted(QUANT_KEY_NAMES)}") from None


# Stop pydantic from introspecting QuantKey: it transitively contains a
# NamedTuple with `ClassVar[GroupShape]` declarations that pydantic refuses.
QuantKeyField = Annotated[
    QuantKey | None,
    GetPydanticSchema(lambda _src, _handler: core_schema.no_info_plain_validator_function(_coerce_quant_key)),
]


@config
class QuantSpec:
    """Quantization spec for one layer kind (linear or MoE).

    `None` on either side means the method class falls back to its own default
    (typically inherited from the checkpoint, or unquantized for online).
    """

    weight: QuantKeyField = None
    """Weight quantization key, or a name from QUANT_KEY_NAMES."""

    activation: QuantKeyField = None
    """Activation quantization key, or a name from QUANT_KEY_NAMES."""

    def __str__(self) -> str:
        if self.weight is None:
            return "None"
        return next((name for name, key in QUANT_KEY_NAMES.items() if key == self.weight), str(self.weight))

    if TYPE_CHECKING:

        def __init__(
            self,
            weight: QuantKeyField = None,
            activation: QuantKeyField = None,
        ) -> None: ...


def _coerce_override_weight(v: Any) -> Any:
    if v in (None, "bf16"):
        return v
    return _coerce_quant_key(v)


OverrideWeightField = Annotated[
    QuantKey | Literal["bf16"] | None,
    GetPydanticSchema(lambda _src, _handler: core_schema.no_info_plain_validator_function(_coerce_override_weight)),
]


@config
class QuantOverride:
    """Ordered module-level override for online quantization."""

    pattern: str = ""
    """Exact module prefix or ``re:`` regular expression."""

    weight: OverrideWeightField = None
    """Replacement weight format; ``bf16`` leaves the module unquantized."""

    activation: QuantKeyField = None
    """Replacement activation format; omitted fields inherit the base spec."""


@config
class QuantizationConfigArgs:
    """User-facing quantization configuration.

    See `docs/features/quantization/online.md` for the schema and shorthand
    string forms accepted on `linear` and `moe`.
    """

    linear: QuantSpec | None = None
    """Spec applied to ``LinearBase`` layers."""

    moe: QuantSpec | None = None
    """Spec applied to ``FusedMoEFactory`` layers."""

    ignore: list[str] = Field(default_factory=list)
    """Layers to skip quantization for. Online quantization also supports
    fnmatch-style patterns."""

    targets: dict[str, str] | None = None
    """Per-layer online quantization overrides, keyed by exact layer name or
    regex patterns with a `re:`, or fnmatch-style patterns for online
    quantization, mapping to an online shorthand name (see
    `_ONLINE_SHORTHANDS`). A layer that matches no pattern is left unquantized.
    Mutually exclusive with `linear` and `moe`.
    """

    overrides: list[QuantOverride] = Field(default_factory=list)
    """Ordered module precision overrides. Later matching rules win, including
    over the defaults supplied by an online shorthand such as ``mxfp6``."""

    if TYPE_CHECKING:

        def __init__(
            self,
            linear: QuantSpec | None = None,
            moe: QuantSpec | None = None,
            ignore: list[str] = ...,
            targets: dict[str, str] | None = None,
            overrides: list[QuantOverride] = ...,
        ) -> None: ...

    @field_validator("linear", "moe", mode="before")
    @classmethod
    def _coerce_spec(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, str):
            return v
        field_name = info.field_name
        assert field_name is not None
        if v in _ONLINE_SHORTHANDS:
            spec = getattr(_ONLINE_SHORTHANDS[v], field_name)
            if spec is None:
                raise ValueError(f"online shorthand {v!r} does not define a {field_name} spec")
            return spec
        return QuantSpec(weight=_coerce_quant_key(v))

    @field_validator("targets", mode="before")
    @classmethod
    def _validate_targets(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, dict):
            raise TypeError(f"targets must be a dict, got {type(v).__name__}")
        for pattern, shorthand in v.items():
            if not isinstance(pattern, str):
                raise ValueError(f"targets keys must be strings, got {type(pattern).__name__}")
            if not isinstance(shorthand, str) or shorthand not in _ONLINE_SHORTHANDS:
                raise ValueError(
                    f"targets[{pattern}] = {shorthand} is not a valid "
                    f"online shorthand name; expected one of "
                    f"{sorted(_ONLINE_SHORTHANDS)}"
                )
            if pattern.startswith("re:"):
                try:
                    re.compile(pattern[3:])
                except re.error as e:
                    raise ValueError(f"targets key {pattern} is not a valid regex: {e}") from e
        return v

    @model_validator(mode="after")
    def _validate_targets_exclusivity(self) -> "QuantizationConfigArgs":
        if self.targets is None:
            return self
        if self.linear is not None or self.moe is not None:
            raise ValueError(
                "quantization_config.targets is mutually exclusive with "
                f"quantization_config.linear/moe, got "
                f"targets={self.targets}, linear={self.linear}, "
                f"moe={self.moe}."
            )
        return self


# CLI shorthands accepted by `--quantization`. Each desugars to a full
# QuantizationConfigArgs; activation overrides go through quantization_config.
_ONLINE_SHORTHANDS: dict[str, QuantizationConfigArgs] = {
    "fp8_per_tensor": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticTensorSym),
        moe=QuantSpec(weight=kFp8StaticTensorSym),
    ),
    "fp8_per_block": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8Static128BlockSym),
        moe=QuantSpec(weight=kFp8Static128BlockSym),
    ),
    # Per-output-channel weight scale + dynamic per-token activation.
    # Same shape as llmcompressor's FP8_DYNAMIC recipe.
    "fp8_per_channel": QuantizationConfigArgs(
        linear=QuantSpec(weight=kFp8StaticChannelSym),
        moe=QuantSpec(weight=kFp8StaticChannelSym),
    ),
    "mxfp8": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp8Dynamic),
        moe=QuantSpec(weight=kMxfp8Dynamic),
    ),
    "mxfp6": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp6E2M3Static, activation=kMxfp8Dynamic),
        moe=QuantSpec(weight=kMxfp6E2M3Static, activation=kMxfp8Dynamic),
        overrides=[
            QuantOverride(
                pattern=r"re:(^|.*\.)(gate|router|shared_expert_gate|lm_head)$",
                weight="bf16",
            )
        ],
    ),
    "mxfp4": QuantizationConfigArgs(
        linear=QuantSpec(weight=kMxfp4Static),
        moe=QuantSpec(weight=kMxfp4Static),
    ),
    # INT8 weight-only on MoE; linear stays unquantized (no `linear` field).
    "int8_per_channel_weight_only": QuantizationConfigArgs(
        moe=QuantSpec(weight=kInt8StaticChannelSym),
    ),
    # Online NVFP4 on MoE with per-token dynamic activation scales (Blackwell +
    # FlashInfer TRTLLM only); linear stays unquantized (no `linear` field).
    "nvfp4_per_token": QuantizationConfigArgs(
        moe=QuantSpec(weight=kNvfp4Static),
    ),
}


# Names accepted by `--quantization`; "online" means "use quantization_config".
ONLINE_QUANT_SHORTHAND_NAMES: tuple[str, ...] = (
    *_ONLINE_SHORTHANDS.keys(),
    "online",
)

# These names are also checkpoint quantization methods. Their online configs
# are resolved only when checkpoint quantization metadata is absent.
_DEFERRED_ONLINE_SHORTHANDS = frozenset(("mxfp4", "mxfp8"))


def resolve_quantization_config(
    quantization: str | None,
    quantization_config: dict[str, Any] | QuantizationConfigArgs | None,
) -> QuantizationConfigArgs | None:
    """Resolve `--quantization` shorthand and `--quantization-config` into a
    QuantizationConfigArgs.

    `quantization` is a CLI shorthand that desugars into a base config via
    `_ONLINE_SHORTHANDS`. `quantization_config` is a dict or pre-built args
    object. When both are given, fields explicitly set in `quantization_config`
    take precedence over the shorthand.
    """
    if quantization is not None and quantization not in ONLINE_QUANT_SHORTHAND_NAMES:
        # Pre-quantized checkpoints can be composed with online quantization
        # for layers that the base quant_method leaves unquantized. The
        # checkpoint quant_method remains the primary quantization method; composition
        # is performed after its config has been loaded.
        if quantization_config is None:
            return None

        # `quantization_config` may hold both:
        # 1. Base quantization method activation key override,
        # 2. online quantization config to apply on top of the base quant_method.
        if isinstance(quantization_config, dict):
            return QuantizationConfigArgs(**quantization_config)
        return quantization_config

    base = _ONLINE_SHORTHANDS.get(quantization) if quantization else None

    if quantization_config is None:
        if quantization in _DEFERRED_ONLINE_SHORTHANDS:
            return None
        return base

    if isinstance(quantization_config, dict):
        quantization_config = QuantizationConfigArgs(**quantization_config)

    if base is None:
        return quantization_config

    return QuantizationConfigArgs(
        linear=quantization_config.linear or base.linear,
        moe=quantization_config.moe or base.moe,
        ignore=quantization_config.ignore or base.ignore,
        overrides=[*base.overrides, *quantization_config.overrides],
        targets=quantization_config.targets or base.targets,
    )
