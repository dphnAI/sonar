# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x modular tensor-parallel fused MoE backend."""

import weakref
from collections.abc import Iterable
from functools import partial
from typing import Any

import torch

import aphrodite.model_executor.layers.fused_moe.modular_kernel as mk
from aphrodite import envs
from aphrodite.model_executor.layers.fused_moe import b12x_fused_decode
from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from aphrodite.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from aphrodite.model_executor.utils import replace_parameter
from aphrodite.platforms import current_platform
from aphrodite.utils.b12x import (
    B12X_BLOCK_CODECS,
    B12xPreparationUnit,
    B12xWarmupUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_fused_moe,
    reuse_packed_weight_storage,
    set_b12x_preparation_provider,
)

_B12X_MOE_MODES: dict[
    tuple[torch.dtype | str | None, torch.dtype | str | None],
    tuple[str, str, str],
] = {
    **{(codec, None): ("a16", codec, "w31") for codec in B12X_BLOCK_CODECS},
    ("mxfp4", "mxfp8"): ("a8", "fp4_e8m0_k32", "w31"),
    ("mxfp4", None): ("a16", "fp4_e8m0_k32", "w31"),
    ("nvfp4", "nvfp4"): ("a4", "modelopt_nvfp4", "w31"),
    ("nvfp4", "mxfp8"): ("a8", "modelopt_nvfp4", "w31"),
    ("nvfp4", None): ("a16", "modelopt_nvfp4", "w13"),
}


def _require_b12x_fused_moe() -> Any:
    fused_moe = get_b12x_fused_moe()
    assert fused_moe is not None
    b12x_fused_decode.install()
    return fused_moe


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation == MoEActivation.SILU:
        return "silu"
    if activation in (MoEActivation.RELU2, MoEActivation.RELU2_NO_MUL):
        return "relu2"
    return activation.value


def _b12x_scratch_nbytes(plan: Any) -> int:
    variants = getattr(plan, "variants", None)
    if variants is not None:
        return max(_b12x_scratch_nbytes(child) for child in variants.values())
    specs = plan.scratch_specs()
    if len(specs) != 1:
        raise RuntimeError(f"expected one b12x MoE scratch buffer, got {len(specs)}")
    spec = specs[0]
    if spec.dtype != torch.uint8:
        raise TypeError(f"expected b12x MoE scratch dtype uint8, got {spec.dtype}")
    return int(spec.shape[0])


def _run_b12x_moe_plan(
    *,
    plan: Any,
    scratch: torch.Tensor,
    hidden_states: torch.Tensor,
    prepared: Any,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    fused_moe = _require_b12x_fused_moe()

    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=hidden_states,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=True,
    )
    fused_moe.run(binding=binding)


@torch.inference_mode()
def _prepare_b12x_moe_plan(
    *,
    plan: Any,
    prepared: Any,
    tokens: int,
    topk: int,
) -> None:
    from b12x.preparation import PreparedCall, prepare_default

    fused_moe = _require_b12x_fused_moe()
    hidden_states = torch.zeros(
        (tokens, int(prepared.hidden_size)),
        dtype=prepared.plan.activation.io_dtype,
        device=prepared.device,
    )
    output = torch.empty_like(hidden_states)
    topk_ids = (
        torch.arange(topk, device=prepared.device, dtype=torch.int32).unsqueeze(0).expand(tokens, -1).contiguous()
    )
    topk_ids.remainder_(int(prepared.num_experts))
    topk_weights = torch.full(
        (tokens, topk),
        1.0 / topk,
        dtype=torch.float32,
        device=prepared.device,
    )
    scratch = torch.empty(
        (_b12x_scratch_nbytes(plan),),
        dtype=torch.uint8,
        device=prepared.device,
    )

    def prepare_call(state: Any) -> PreparedCall:
        binding = state.bind(
            scratch=scratch,
            a=hidden_states,
            experts=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=True,
        )
        # Priming must exercise the retained launchers. Do not retain these
        # temporary tensors: serving supplies the shared caller-owned scratch.
        return PreparedCall(run=lambda: fused_moe.run(binding=binding))

    prepare_default(plan.request(name="aphrodite:moe", prepare_call=prepare_call))


def _prepared_moe_call_factory(*, tokens: int, topk: int, prepared: Any, output_dtype: torch.dtype) -> Any:
    """Build bounded trial bindings without retaining serving scratch."""
    shared = None

    def factory(state: Any) -> Any:
        from b12x.moe.fused_moe.workloads import make_tuning_routes
        from b12x.preparation import PreparedCall

        nonlocal shared
        tensors = None if shared is None else tuple(reference() for reference in shared)
        if tensors is None or any(tensor is None for tensor in tensors):
            device = prepared.device
            hidden = torch.empty(
                (tokens, int(prepared.hidden_size)),
                dtype=prepared.plan.activation.io_dtype,
                device=device,
            )
            generator = torch.Generator(device=device).manual_seed(42)
            activation_source = torch.empty_like(hidden).normal_(mean=0.0, std=0.125, generator=generator)
            output = torch.empty(hidden.shape, dtype=output_dtype, device=device)
            route_ids = make_tuning_routes(tokens, topk, int(prepared.num_experts), device=device)
            rows = torch.arange(tokens, dtype=torch.float32, device=device).unsqueeze(1)
            columns = torch.arange(topk, dtype=torch.float32, device=device).unsqueeze(0)
            route_weights = torch.softmax(rows * 0.03125 + columns * 0.125, dim=-1).contiguous()
            ids = torch.empty_like(route_ids[0])
            weights = torch.empty_like(route_weights)
            tensors = (hidden, activation_source, output, route_ids, route_weights, ids, weights)
            shared = tuple(weakref.ref(tensor) for tensor in tensors)
        hidden, activation_source, output, route_ids, route_weights, ids, weights = tensors
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) for spec in state.scratch.scratch_specs()
        )

        def reset() -> None:
            output.zero_()
            for buffer in scratch:
                buffer.zero_()

        def produce(pattern: int = 0) -> None:
            hidden.copy_(activation_source)
            ids.copy_(route_ids[pattern])
            weights.copy_(route_weights)

        def restore() -> None:
            reset()
            produce()

        binding = state.bind(
            scratch=scratch,
            a=hidden,
            experts=prepared,
            topk_weights=weights,
            topk_ids=ids,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=produce,
            reset=reset,
            restore=restore,
            capture_safe=False,
            benchmark_producers=tuple(partial(produce, pattern) for pattern in range(route_ids.shape[0])),
        )

    return factory


def _is_current_stream_capturing() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


def _normalize_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
        return topk_ids
    if _is_current_stream_capturing():
        raise RuntimeError("b12x MoE topk_ids normalization would allocate during CUDA capture")
    return topk_ids.to(dtype=torch.int32).contiguous()


def _normalize_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous():
        return topk_weights
    if _is_current_stream_capturing():
        raise RuntimeError("b12x MoE topk_weights normalization would allocate during CUDA capture")
    return topk_weights.to(dtype=torch.float32).contiguous()


def _workspace_as_b12x_scratch(
    workspace: torch.Tensor | None,
    plan: Any,
    *,
    required_nbytes: int | None = None,
) -> torch.Tensor:
    if workspace is None:
        raise RuntimeError("b12x MoE requires workspace2 scratch")
    if not workspace.is_contiguous():
        raise ValueError("b12x MoE workspace2 must be contiguous")
    scratch = workspace.view(-1).view(torch.uint8)
    if required_nbytes is None:
        required_nbytes = _b12x_scratch_nbytes(plan)
    if scratch.numel() < required_nbytes:
        raise ValueError(
            f"b12x MoE workspace2 is too small: have={scratch.numel()} bytes, need={required_nbytes} bytes"
        )
    return scratch


def _replace_parameter_with_empty(
    layer: torch.nn.Module,
    name: str,
) -> torch.Tensor | None:
    parameter = getattr(layer, name, None)
    if not isinstance(parameter, torch.Tensor):
        return None
    empty = torch.empty((0,), dtype=parameter.dtype, device=parameter.device)
    replace_parameter(layer, name, empty)
    return getattr(layer, name)


def _normalize_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        if scale.shape[1] not in (1, 2):
            raise ValueError(f"expected an expert scale with one or two columns, got {tuple(scale.shape)}")
        scale = scale[:, 0]
    return scale.to(dtype=torch.float32).contiguous()


def _canonicalize_fp4_zero_signs_(packed: torch.Tensor) -> None:
    """Clear sign bits from packed FP4 zero values in place."""
    packed = packed.view(torch.uint8)
    magnitude = packed & 0x77
    nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11
    packed.bitwise_and_(0x77 | (nonzero << 3))


class B12xExperts(mk.FusedMoEExpertsModular):
    """FP4 MoE experts backed by the b12x SM12x planned API."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        if quant_config.weight_quant_dtype not in (
            "mxfp4",
            "nvfp4",
            *B12X_BLOCK_CODECS,
        ):
            raise ValueError(f"unsupported b12x MoE weight format: {quant_config.weight_quant_dtype}")
        scheme = (
            quant_config.weight_quant_dtype,
            quant_config.quant_dtype,
        )
        try:
            self._activation_mode, self._source_format, self._w13_layout = _B12X_MOE_MODES[scheme]
        except KeyError as exc:
            raise ValueError(f"unsupported b12x MoE quantization scheme {scheme}") from exc
        self._prepared_experts: Any | None = None
        self._source_parameters_released = False
        self._unit_scales: dict[torch.device, torch.Tensor] = {}
        self._plans: dict[tuple[int, int, MoEActivation, bool], Any] = {}
        self._workspace_nbytes = 0
        self._max_tokens = int(moe_config.max_num_tokens)
        self._warmup_token_counts = {1}
        self._apply_router_weight_on_input = False
        self._native_preparation = False
        if (
            envs.APHRODITE_DSV41_SM120_NATIVE
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            from aphrodite.config import get_current_aphrodite_config

            self._native_preparation = get_current_aphrodite_config().model_config.quantization == "deepseek_v41_fp8"
        self._native_plan: Any | None = None
        self._native_plan_key: tuple[Any, ...] | None = None
        self._native_scratch_owner: Any | None = None

    def _unit_scale(self, device: torch.device, num_experts: int) -> torch.Tensor:
        scale = self._unit_scales.get(device)
        if scale is None or scale.numel() != num_experts:
            scale = torch.ones(num_experts, dtype=torch.float32, device=device)
            self._unit_scales[device] = scale
        return scale

    def _weight_global_scale(
        self,
        device: torch.device,
        num_experts: int,
        scale: torch.Tensor | None,
        name: str,
    ) -> torch.Tensor:
        if self._source_format != "modelopt_nvfp4":
            return self._unit_scale(device, num_experts)
        if scale is None:
            raise ValueError(f"b12x NVFP4 MoE requires {name}")
        scale = _normalize_expert_scale(scale)
        if scale.numel() != num_experts:
            raise ValueError(f"b12x NVFP4 MoE expected {num_experts} {name} values, got {scale.numel()}")
        return scale.to(device=device)

    def _swiglu_params(
        self,
        activation: MoEActivation,
    ) -> tuple[float | None, float | None, float | None]:
        if activation in (
            MoEActivation.SITU,
            MoEActivation.RELU2,
            MoEActivation.RELU2_NO_MUL,
        ):
            return None, None, None

        limit = self.quant_config.gemm1_clamp_limit
        if limit is None:
            limit = self.moe_config.swiglu_limit
        if activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return limit, None, None

        alpha = self.quant_config.gemm1_alpha
        if alpha is None:
            alpha = self.moe_config.swiglu_alpha
        beta = self.quant_config.gemm1_beta
        if beta is None:
            beta = self.moe_config.swiglu_beta
        return limit, alpha, beta

    @torch.inference_mode()
    def _prepare_experts(
        self,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        params_dtype: torch.dtype,
    ) -> Any:
        if _is_current_stream_capturing():
            raise RuntimeError("b12x MoE weights must be prepared before CUDA graph capture")
        fused_moe = _require_b12x_fused_moe()
        block_size = B12X_BLOCK_CODECS[self._source_format][0] if self._source_format in B12X_BLOCK_CODECS else 2
        limit, alpha, beta = self._swiglu_params(activation)
        weight_plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format=self._source_format, w13_layout=self._w13_layout),
            activation=fused_moe.ActivationSpec(
                mode=self._activation_mode,
                nonlinearity=_b12x_activation_name(activation),
                io_dtype=params_dtype,
                swiglu_limit=limit,
                swiglu_alpha=alpha,
                swiglu_beta=beta,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=int(w1.shape[0]),
                hidden_size=int(w2.shape[1]),
                intermediate_size=int(w2.shape[2]) * block_size,
            ),
        )
        if self._source_format in B12X_BLOCK_CODECS:
            return fused_moe.prepare_weights(
                plan=weight_plan,
                weights=fused_moe.BlockQuantWeights(w13=w1, w2=w2, codec=self._source_format),
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise ValueError("b12x MoE requires w1 and w2 block scales")

        _canonicalize_fp4_zero_signs_(w1)
        _canonicalize_fp4_zero_signs_(w2)

        num_experts = int(w1.shape[0])
        unit_scale = self._unit_scale(w1.device, num_experts)
        w1_global_scale = self._weight_global_scale(w1.device, num_experts, self.g1_alphas, "w1 global scales")
        w2_global_scale = self._weight_global_scale(w2.device, num_experts, self.g2_alphas, "w2 global scales")

        if self._source_format == "modelopt_nvfp4" and self._activation_mode != "a16":
            if self.a1_gscale is None or self.a2_gscale is None:
                raise ValueError("b12x NVFP4 MoE requires activation global scales")
            # Own immutable inference tensors so compatible reloads can copy
            # scales without invalidating a versioned native sharing proof.
            a1_gscale = _normalize_expert_scale(self.a1_gscale).to(device=w1.device, copy=True)
            a2_gscale = _normalize_expert_scale(self.a2_gscale).to(device=w2.device, copy=True)
        else:
            a1_gscale = unit_scale
            a2_gscale = unit_scale

        return fused_moe.prepare_weights(
            plan=weight_plan,
            weights=fused_moe.PackedWeights(
                w13=w1,
                w2=w2,
                w13_block_scales=self.w1_scale,
                w2_block_scales=self.w2_scale,
                w13_global_scales=w1_global_scale,
                w2_global_scales=w2_global_scale,
                input_scale=a1_gscale,
                intermediate_scale=a2_gscale,
                immutable_input_scales=True,
            ),
        )

    def _refresh_quant_config(self, layer: torch.nn.Module) -> None:
        if self._source_format in B12X_BLOCK_CODECS:
            return
        self.quant_config._w1.scale = layer.w13_weight_scale
        self.quant_config._w2.scale = layer.w2_weight_scale
        if self._source_format != "modelopt_nvfp4":
            return

        self.quant_config._w1.alpha_or_gscale = layer.w13_weight_scale_2
        self.quant_config._w2.alpha_or_gscale = layer.w2_weight_scale_2
        if self._activation_mode != "a16":
            self.quant_config._a1.alpha_or_gscale = 1.0 / layer.w13_input_scale
            self.quant_config._a2.alpha_or_gscale = 1.0 / layer.w2_input_scale

    def _release_source_parameters(self, layer: torch.nn.Module) -> None:
        if self._source_parameters_released:
            return
        w1_scale = _replace_parameter_with_empty(layer, "w13_weight_scale")
        w2_scale = _replace_parameter_with_empty(layer, "w2_weight_scale")
        if w1_scale is not None:
            self.quant_config._w1.scale = w1_scale
        if w2_scale is not None:
            self.quant_config._w2.scale = w2_scale
        _replace_parameter_with_empty(layer, "w13_weight")
        _replace_parameter_with_empty(layer, "w2_weight")
        self._source_parameters_released = True

    @torch.inference_mode()
    def _reuse_prepared_storage(self, layer: torch.nn.Module, prepared: Any) -> Any:
        previous = getattr(layer, "_b12x_prepared_experts", None)
        prepared = reuse_packed_weight_storage(previous, prepared)
        if prepared is not previous:
            self._plans.clear()
            self._workspace_nbytes = 0
            self._native_plan = None
            self._native_plan_key = None
            self._native_scratch_owner = None
        self._prepared_experts = prepared
        layer._b12x_prepared_experts = prepared
        return prepared

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._apply_router_weight_on_input = layer.apply_router_weight_on_input
        if self._apply_router_weight_on_input and self._activation_mode != "a16":
            raise ValueError("b12x MoE supports apply_router_weight_on_input only with W4A16")
        self._source_parameters_released = False
        self._refresh_quant_config(layer)
        prepared = self._prepare_experts(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            activation=layer.activation,
            params_dtype=self.moe_config.in_dtype,
        )
        prepared = self._reuse_prepared_storage(layer, prepared)
        if prepared._impl.plan.discards_source_parameters:
            self._release_source_parameters(layer)
        layer.b12x_warmup_provider = self
        if self._native_preparation:
            set_b12x_preparation_provider(layer, self)

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.has_bias:
            return False, "kernel does not support expert biases"
        if moe_config.in_dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )
        if moe_config.activation == MoEActivation.SITU and (
            moe_config.activation_situ_beta != 4.0 or moe_config.activation_situ_linear_beta != 25.0
        ):
            return False, "kernel supports only SiTU beta=4 and linear_beta=25"
        if activation_key is not None and moe_config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return (
                False,
                "kernel does not support swigluoai_uninterleave with W4A8",
            )
        unpadded_intermediate_size = (
            moe_config.intermediate_size_per_partition_unpadded or moe_config.intermediate_size_per_partition
        )
        if weight_key == kMxfp4Static and unpadded_intermediate_size % 32 != 0:
            return (
                False,
                "MXFP4 requires the per-rank intermediate size to be divisible by 32",
            )
        if weight_key == kMxfp4Static and activation_key == kMxfp8Dynamic:
            if moe_config.activation not in (
                MoEActivation.SILU,
                MoEActivation.SITU,
            ):
                return False, "MXFP4 W4A8 supports only SiLU and SiTU"
            if moe_config.hidden_dim % 256 != 0 or moe_config.intermediate_size_per_partition % 32 != 0:
                return (
                    False,
                    ("MXFP4 W4A8 requires hidden size divisible by 256 and per-rank intermediate size divisible by 32"),
                )
        return mk.FusedMoEExperts.is_supported_config(cls, moe_config, weight_key, activation_key, activation_format)

    @staticmethod
    def _supports_current_device() -> bool:
        if not (current_platform.is_cuda() and current_platform.is_device_capability_family(120)):
            return False
        fused_moe = get_b12x_fused_moe()
        if fused_moe is None:
            return False
        return fused_moe.is_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kMxfp4Static, kMxfp8Dynamic),
            (kMxfp4Static, None),
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, kMxfp8Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SITU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size == 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _prepared(self) -> Any:
        if self._prepared_experts is None:
            raise RuntimeError("b12x MoE weights must be prepared by process_weights_after_loading")
        return self._prepared_experts

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if w1.numel() and w2.numel() and self._source_format not in B12X_BLOCK_CODECS:
            return super().moe_problem_size(a1, w1, w2, topk_ids)
        prepared = self._prepared()
        tokens = int(a1.shape[0] if a1.ndim == 2 else a1.shape[1])
        return (
            int(prepared.num_experts),
            tokens,
            int(prepared.intermediate_size) * (2 if self.moe_config.activation.is_gated else 1),
            int(a1.shape[-1]),
            int(topk_ids.shape[1]),
        )

    def _plan(
        self,
        *,
        tokens: int,
        topk: int,
        activation: MoEActivation,
        apply_router_weight_on_input: bool = False,
    ) -> Any:
        if self._native_preparation:
            return self._prepared_plan(
                tokens=tokens,
                topk=topk,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        fused_moe = _require_b12x_fused_moe()

        tokens = max(int(tokens), 1)
        if tokens > self._max_tokens:
            raise ValueError(f"b12x MoE token count {tokens} exceeds prepared capacity {self._max_tokens}")
        # The capacity plan also handles undeclared partial prefills, exactly
        # like the maximum-capacity child of a native composite declaration.
        planned_tokens = tokens if tokens in self._warmup_token_counts else self._max_tokens
        key = (
            planned_tokens,
            int(topk),
            activation,
            bool(apply_router_weight_on_input),
        )
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        if _is_current_stream_capturing():
            raise RuntimeError("b12x MoE plans must be created before CUDA capture")

        prepared = self._prepared()
        if _b12x_activation_name(activation) != prepared.plan.activation.nonlinearity:
            raise ValueError("activation does not match the prepared b12x MoE weights")
        plan = fused_moe.plan_execution(
            experts=prepared,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=key[0],
                top_k=key[1],
                warmup_token_counts=(key[0],),
                route_num_experts=0,
            ),
            routing=fused_moe.RoutingSpec(apply_router_weight_on_input=key[3]),
        )
        _prepare_b12x_moe_plan(
            plan=plan,
            prepared=prepared,
            tokens=key[0],
            topk=key[1],
        )
        # Graph variants and partial prefills share one fixed runtime arena.
        self._workspace_nbytes = max(self._workspace_nbytes, _b12x_scratch_nbytes(plan))
        self._plans[key] = plan
        return plan

    def _prepared_plan(
        self,
        *,
        tokens: int,
        topk: int,
        activation: MoEActivation,
        apply_router_weight_on_input: bool,
    ) -> Any:
        plan = self._native_plan
        key = self._native_plan_key
        payload = None if plan is None else plan.prepared
        if (
            plan is None
            or key is None
            or key[1:4] != (int(topk), activation, bool(apply_router_weight_on_input))
            or key[4] != id(self._prepared())
            or payload is None
        ):
            raise PreparationResourceUnavailableError("b12x MoE requires a prepared native activation/routing plan")
        if not 0 < int(tokens) <= key[0][-1]:
            raise ValueError(f"b12x MoE token count {tokens} exceeds prepared capacity {key[0][-1]}")
        if self._native_scratch_owner is None or self._native_scratch_owner() is not payload:
            # Size every winning child, including equal-alignment scratch ties.
            self._workspace_nbytes = _b12x_scratch_nbytes(plan)
            self._native_scratch_owner = weakref.ref(payload)
        return plan

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if not self._native_preparation or workload.stage != "weights":
            return ()
        from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION
        from b12x.preparation import FrozenMapping

        if workload.output_dtype != self.moe_config.in_dtype:
            raise ValueError("b12x MoE output dtype differs from its loaded contract")
        if _is_current_stream_capturing():
            raise PreparationResourceUnavailableError("b12x MoE declaration is forbidden during CUDA capture")
        prepared = self._prepared()
        activation = layer.activation
        if _b12x_activation_name(activation) != prepared.plan.activation.nonlinearity:
            raise ValueError("activation does not match the prepared b12x MoE weights")
        counts = tuple(sorted({workload.max_tokens, *workload.fixed_token_counts}))
        topk = int(self.moe_config.experts_per_token)
        route_on_input = bool(layer.apply_router_weight_on_input)
        key = (counts, topk, activation, route_on_input, id(prepared))
        plan = self._native_plan if self._native_plan_key == key else None
        if plan is None:
            fused_moe = _require_b12x_fused_moe()
            plan = fused_moe.plan_execution(
                experts=prepared,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=workload.max_tokens,
                    top_k=topk,
                    warmup_token_counts=counts,
                    route_num_experts=0,
                ),
                routing=fused_moe.RoutingSpec(apply_router_weight_on_input=route_on_input),
                invocation=FrozenMapping({"tuning_route_pattern": TUNING_WORKLOAD_VERSION}),
            )
            self._native_plan = plan
            self._native_plan_key = key
        name = f"fused_moe:{id(layer)}"

        def call(count: int) -> Any:
            return _prepared_moe_call_factory(
                tokens=count, topk=topk, prepared=prepared, output_dtype=workload.output_dtype
            )

        if hasattr(plan, "token_counts"):
            request = plan.request(
                name=name,
                prepare_calls={count: call(count) for count in plan.token_counts},
                benchmark_calls={count: call(count) for count in plan.token_counts},
            )
        else:
            request = plan.request(name=name, prepare_call=call(counts[0]), benchmark_call=call(counts[0]))
        return (
            B12xPreparationUnit(
                name="MoE",
                key=(self._activation_mode, self._source_format, self._w13_layout, key, workload.output_dtype),
                requests=(request,),
                stage="weights",
            ),
        )

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        assert output_dtype == self.moe_config.in_dtype
        prepared = self._prepared()

        def compile() -> None:
            self.warmup_launches(layer, token_counts=token_counts)

        return B12xWarmupUnit(
            name="MoE",
            # Canonical plans retain this expert owner, not just its geometry.
            # Native program caches still deduplicate compilation across layers.
            key=(type(self), id(prepared)),
            compile=compile,
        )

    @torch.inference_mode()
    def warmup_launches(
        self,
        layer: torch.nn.Module,
        *,
        token_counts: Iterable[int],
    ) -> int:
        """Prime the prefill capacity and every declared exact specialization."""
        counts = {1, *(int(count) for count in token_counts if int(count) > 0)}
        if self._native_preparation:
            # Both target and draft still visit legacy warmup, but the session
            # has already selected and primed their combined capacity plans.
            self._prepared_plan(
                tokens=max(counts),
                topk=int(self.moe_config.experts_per_token),
                activation=layer.activation,
                apply_router_weight_on_input=bool(layer.apply_router_weight_on_input),
            )
            assert self._native_plan_key is not None
            return len(self._native_plan_key[0])
        self._warmup_token_counts.update(counts)
        self._max_tokens = max(self._max_tokens, *counts)
        planned_counts = sorted({self._max_tokens, *self._warmup_token_counts})
        for tokens in planned_counts:
            self._plan(
                tokens=tokens,
                topk=int(self.moe_config.experts_per_token),
                activation=layer.activation,
                apply_router_weight_on_input=bool(layer.apply_router_weight_on_input),
            )
        return len(planned_counts)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, global_num_experts, local_num_experts, expert_tokens_meta
        self._plan(
            tokens=M,
            topk=topk,
            activation=activation,
            apply_router_weight_on_input=self._apply_router_weight_on_input,
        )
        itemsize = self.moe_config.in_dtype.itemsize
        scratch_elements = max(1, (self._workspace_nbytes + itemsize - 1) // itemsize)
        return (0,), (scratch_elements,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del w1, w2, global_num_experts
        del a1q_scale, a2_scale, workspace13, expert_tokens_meta
        if expert_map is not None:
            raise ValueError("b12x TP MoE does not support expert maps")
        if bool(apply_router_weight_on_input) != self._apply_router_weight_on_input:
            raise ValueError("apply_router_weight_on_input does not match the prepared b12x MoE plan")
        prepared = self._prepared()
        topk_ids = _normalize_topk_ids(topk_ids)
        topk_weights = _normalize_topk_weights(topk_weights)
        plan = self._plan(
            tokens=int(hidden_states.shape[0]),
            topk=int(topk_ids.shape[1]),
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
        )
        scratch = _workspace_as_b12x_scratch(
            workspace2, plan, required_nbytes=self._workspace_nbytes if self._native_preparation else None
        )

        _run_b12x_moe_plan(
            plan=plan,
            scratch=scratch,
            hidden_states=hidden_states,
            prepared=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")
