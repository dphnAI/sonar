# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native b12x execution boundaries for the V4.1 checkpoint contract."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from functools import cache
from weakref import WeakValueDictionary

import torch
from b12x.gemm import bf16_gemv, block_fp8_linear
from b12x.norm import hyperconnection, mhc
from b12x.preparation import FrozenMapping, Plan, PreparedCall, plan_from_handle
from torch import nn

import aphrodite.envs as envs
from aphrodite.config import get_current_aphrodite_config
from aphrodite.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from aphrodite.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
)
from aphrodite.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from aphrodite.model_executor.parameter import BlockQuantScaleParameter
from aphrodite.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    b12x_layer,
    set_b12x_preparation_provider,
)
from aphrodite.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    aux_stream,
    current_stream,
)
from aphrodite.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)


def _capacity() -> int:
    return get_current_aphrodite_config().scheduler_config.max_num_batched_tokens


def _execution_capacities() -> tuple[int, ...]:
    config = get_current_aphrodite_config()
    capacity = _capacity()
    spec = config.speculative_config
    decode_capacity = min(
        capacity,
        max(
            config.scheduler_config.max_num_seqs * (1 + (spec.num_speculative_tokens if spec is not None else 0)),
            config.compilation_config.max_cudagraph_capture_size or 0,
        ),
    )
    graph_sizes = config.compilation_config.cudagraph_capture_sizes or ()
    from .ced import ced_decoder_start

    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    ced_capacities: Iterable[int] = ()
    if hf is not None and ced_decoder_start(hf) is not None:
        window = hf.sliding_window
        ced_capacities = range(
            window,
            min(capacity, config.scheduler_config.max_num_seqs * window) + 1,
            window,
        )
    return tuple(
        sorted(
            {
                capacity,
                decode_capacity,
                *(size for size in graph_sizes if 0 < size <= decode_capacity),
                *ced_capacities,
            }
        )
    )


_LINEARS: WeakValueDictionary[int, nn.Module] = WeakValueDictionary()


@cache
def _mhc_caps(device, capacity, hidden):
    return mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden, split_k=hidden // 64)


def block32_linear_plan(layer: nn.Module, rows: int) -> Plan:
    """Select the original block32 declaration for the live row count."""
    index = bisect_left(layer.b12x_capacities, rows)
    if index == len(layer.b12x_capacities):
        raise ValueError("V4.1 linear rows exceed planned capacity")
    return layer.b12x_plans[index]


def block32_linear_prequantized(
    layer: nn.Module, plan: Plan, values: torch.Tensor, scale_mma: torch.Tensor
) -> torch.Tensor:
    """Run the original prepared dense projection without quantizing Q again."""
    prepared = plan.prepared
    packed = getattr(layer, "b12x_weight", None)
    if prepared is None or packed is None or prepared.state.dense is None:
        raise PreparationResourceUnavailableError("V4.1 prequantized block32 projection is not prepared")
    if getattr(layer, "bias", None) is not None:
        raise ValueError("V4.1 block32 projections require bias-free weights")
    state = prepared.state
    query = state.query
    if query.source_dtype != "bfloat16" or query.output_dtype != "bfloat16" or query.output_mode != "provided":
        raise ValueError("V4.1 prequantized block32 projection requires the original BF16 output contract")
    rows = values.shape[0]
    output = torch.empty((rows, query.out_features, 1), dtype=torch.bfloat16, device=state.device)
    workspace = None
    if state.scratch.workspace_nbytes:
        specs = (((state.scratch.workspace_nbytes // 4,), torch.float32),)
        manager = current_workspace_manager()
        stream = current_stream()
        buffers = (
            manager.get_stream_simultaneous(stream.stream_id, *specs)
            if stream == aux_stream()
            else manager.get_simultaneous(*specs)
        )
        retain_cuda_graph_capture_resource(buffers)
        workspace = buffers[0]
    state.dense.run(
        (values.view(rows, query.in_features, 1), scale_mma),
        (packed.weight.values.view(query.out_features, query.in_features, 1), packed.weight.scale_mma),
        out=output,
        split_k_workspace=workspace,
    )
    return output[:, :, 0]


@torch.library.custom_op("aphrodite::dsv41_block32_linear", mutates_args=("out",))
def _block32_linear(x: torch.Tensor, out: torch.Tensor, key: int) -> None:
    layer = _LINEARS[key]
    rows = x.numel() // layer.weight.shape[1]
    plan = block32_linear_plan(layer, rows)
    specs = tuple((spec.shape, spec.dtype) for spec in plan.scratch_specs())
    manager = current_workspace_manager()
    stream = current_stream()
    if stream == aux_stream():
        buffers = manager.get_stream_simultaneous(stream.stream_id, *specs)
    else:
        buffers = manager.get_simultaneous(*specs)
    binding = block_fp8_linear.bind(
        plan,
        scratch=buffers,
        source=x,
        packed_weight=layer.b12x_weight,
        output=out.view(-1, layer.weight.shape[0], 1),
    )
    # Source and output have ordinary caller-managed Tensor lifetimes. Keep
    # the borrowed workspace owner, not every layer's activation storage.
    retain_cuda_graph_capture_resource(buffers)
    block_fp8_linear.run(binding=binding)


@_block32_linear.register_fake
def _block32_linear_fake(x, out, key):
    return None


class B12xEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Retain sharded weight loading/ties; replace only token-row compute."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.b12x_embedding_plans = {}
        set_b12x_preparation_provider(layer, self)

    def get_b12x_preparation_units(self, layer: nn.Module, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        from b12x.sequence import embedding

        if workload.stage != "weights" or layer.weight.is_meta:
            return ()
        weight = layer.weight

        def make_call(state):
            ids = torch.empty(
                state.query.max_rows,
                device=weight.device,
                dtype=getattr(torch, state.query.id_dtype),
            )
            out = torch.empty(
                (state.query.max_rows, weight.shape[1]),
                device=weight.device,
                dtype=weight.dtype,
            )
            return PreparedCall(
                run=lambda: state.run(weight, ids, out=out),
                produce=lambda: ids.copy_(
                    torch.arange(ids.numel(), device=ids.device, dtype=ids.dtype).remainder_(weight.shape[0])
                ),
                owners=(weight,),
            )

        requests = []
        for id_dtype in (torch.int32, torch.int64):
            plan = layer.b12x_embedding_plans.get(id_dtype)
            if plan is None:
                query = embedding.EmbeddingQuery(
                    max_rows=workload.max_tokens,
                    table_rows=weight.shape[0],
                    width=weight.shape[1],
                    row_stride=weight.stride(0),
                    weight_dtype=str(weight.dtype).removeprefix("torch."),
                    id_dtype=str(id_dtype).removeprefix("torch."),
                )
                plan = embedding.plan(query, device=weight.device)
                layer.b12x_embedding_plans[id_dtype] = plan
            requests.append(
                plan.request(
                    name=f"deepseek_v41.embedding.{id(layer):x}.{id_dtype}",
                    prepare_call=make_call,
                )
            )
        return (
            B12xPreparationUnit(
                name="V41Embedding",
                key=(id(layer), workload.max_tokens),
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        from b12x.sequence import embedding

        # Each invocation owns its result: DSpark retains earlier Markov rows
        # for confidence evaluation. Graph capture owns these fixed allocations.
        out = torch.empty(
            (*input_.shape, layer.weight.shape[1]),
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        embedding.run(
            layer.weight,
            input_,
            out=out,
            plan=layer.b12x_embedding_plans[input_.dtype],
        )
        return out


class B12xLinearMethod(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.b12x_linear_plans = {}
        set_b12x_preparation_provider(layer, self)

    def get_b12x_preparation_units(self, layer: nn.Module, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        if workload.stage != "weights" or layer.weight.is_meta:
            return ()
        weight = layer.weight
        capacities = tuple(sorted({workload.max_tokens, *workload.token_counts}))
        layer.b12x_linear_capacity = workload.max_tokens

        def make_call(state):
            query = state.query
            source = torch.empty(
                (query.max_rows, query.in_features),
                device=weight.device,
                dtype=getattr(torch, query.source_dtype),
            )
            out = torch.empty(
                (query.max_rows, query.out_features),
                device=weight.device,
                dtype=getattr(torch, query.output_dtype),
            )
            return PreparedCall(
                run=lambda: state.run(source, weight, out=out),
                produce=lambda: source.normal_(std=0.25),
                owners=(weight,),
            )

        requests = []
        for capacity in capacities:
            for dtype in (torch.bfloat16, torch.float32):
                key = (dtype, capacity)
                plan = layer.b12x_linear_plans.get(key)
                if plan is None:
                    query = bf16_gemv.GemvQuery(
                        source_dtype=str(dtype).removeprefix("torch."),
                        weight_dtype=str(weight.dtype).removeprefix("torch."),
                        output_dtype=str(getattr(layer, "out_dtype", torch.bfloat16)).removeprefix("torch."),
                        max_rows=capacity,
                        in_features=weight.shape[1],
                        out_features=weight.shape[0],
                        source_contiguous=True,
                        source_aligned=True,
                        weight_contiguous=weight.is_contiguous(),
                        weight_aligned=weight.data_ptr() % 16 == 0,
                    )
                    plan = bf16_gemv.plan(query)
                    layer.b12x_linear_plans[key] = plan
                requests.append(
                    plan.request(
                        name=f"deepseek_v41.linear.{id(layer):x}.{dtype}.m{capacity}",
                        prepare_call=make_call,
                        benchmark_call=make_call,
                    )
                )
        return (
            B12xPreparationUnit(
                name="V41Linear",
                key=(id(layer), capacities),
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if bias is not None:
            raise ValueError("V4.1 native projections require bias-free weights")
        plan = layer.b12x_linear_plans.get((x.dtype, x.shape[0]))
        if plan is None:
            plan = layer.b12x_linear_plans[(x.dtype, layer.b12x_linear_capacity)]
        return bf16_gemv.mm(
            x,
            layer.weight,
            plan=plan,
            output_dtype=getattr(layer, "out_dtype", torch.bfloat16),
        )


class B12xFP8LinearMethod(LinearMethodBase):
    """Checkpoint block32 FP8, without a foreign kernel selection phase."""

    def __init__(self, quant_config, *, shared_expert: bool = False):
        if quant_config.weight_block_size != [32, 32]:
            raise ValueError("V4.1 requires checkpoint block32 FP8 linears")
        self._shared_expert = shared_expert

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        layer.weight_block_size = [32, 32]
        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(sum(output_partition_sizes), input_size_per_partition, loader),
        )
        layer.register_parameter(
            "weight_scale_inv",
            create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                [32, 32],
                loader,
                scale_dtype=torch.float8_e8m0fnu,
            ),
        )

    def process_weights_after_loading(self, layer):
        layer.b12x_weight = block_fp8_linear.pack_weight(layer.weight, layer.weight_scale_inv, block_size=(32, 32))
        layer.b12x_capacities = _execution_capacities()
        layer.b12x_plans = tuple(
            block_fp8_linear.plan(
                block_fp8_linear.Caps(
                    device=layer.weight.device,
                    max_tokens=capacity,
                    in_features=layer.weight.shape[1],
                    out_features=layer.weight.shape[0],
                    block_size=(32, 32),
                )
            )
            for capacity in layer.b12x_capacities
        )
        layer.b12x_key = id(layer)
        _LINEARS[layer.b12x_key] = layer
        set_b12x_preparation_provider(layer, self)

    def get_b12x_preparation_units(self, layer: nn.Module, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        if workload.stage != "weights" or layer.weight.is_meta:
            return ()

        def make_call(state):
            query = state.query
            source = torch.empty(
                (query.max_tokens, query.in_features),
                device=state.device,
                dtype=getattr(torch, query.source_dtype),
            )
            output = torch.empty(
                (query.max_tokens, query.out_features, 1),
                device=state.device,
                dtype=getattr(torch, query.output_dtype),
            )
            scratch = [
                torch.empty(spec.shape, dtype=spec.dtype, device=state.device) for spec in state.scratch.scratch_specs()
            ]
            binding = state.bind(
                scratch=scratch,
                source=source,
                packed_weight=layer.b12x_weight,
                output=output,
            )
            return PreparedCall(
                run=lambda: state.run_binding(binding),
                produce=lambda: source.normal_(std=0.25),
                owners=(layer.b12x_weight,),
            )

        aux_capacity = 0
        threshold = envs.APHRODITE_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        if self._shared_expert and threshold > 0:
            index = bisect_left(layer.b12x_capacities, min(threshold, layer.b12x_capacities[-1]))
            aux_capacity = layer.b12x_capacities[index]

        requests = tuple(
            plan.request(
                name=(
                    f"deepseek_v41.shared_block32.{id(layer):x}.m{capacity}"
                    if capacity <= aux_capacity
                    else f"deepseek_v41.block32.{id(layer):x}.m{capacity}"
                ),
                prepare_call=make_call,
                benchmark_call=make_call,
            )
            for capacity, plan in zip(layer.b12x_capacities, layer.b12x_plans)
        )
        return (
            B12xPreparationUnit(
                name="V41Block32Linear",
                key=(id(layer), layer.b12x_capacities),
                requests=requests,
                stage="weights",
                auxiliary_stream_requests=frozenset(
                    request.name
                    for capacity, request in zip(layer.b12x_capacities, requests)
                    if capacity <= aux_capacity
                ),
            ),
        )

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("V4.1 block32 projections require bias-free weights")
        out = torch.empty(
            (*x.shape[:-1], layer.weight.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        _block32_linear(x, out, layer.b12x_key)
        return out


@torch.library.custom_op("aphrodite::dsv41_rmsnorm", mutates_args=("out",))
def _rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    p = plan_from_handle(plan_handle)
    # These unused binding slots must remain disjoint from normalized storage.
    binding = hyperconnection.bind(
        p,
        normalized=out,
        bottleneck=x.view(-1)[: x.shape[0]].view(-1, 1),
        block_input=x.view(-1)[: x.shape[0] * p.query.hidden_size].view(-1, p.query.hidden_size),
        tokens=x.shape[0],
    )
    hyperconnection.run_grouped_rmsnorm(x, weight, eps=eps, binding=binding, zero_centered=False)


@_rmsnorm.register_fake
def _rmsnorm_fake(x, weight, out, eps, plan_handle):
    return None


class B12xRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, streams: int = 1):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size * streams, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.variance_epsilon = eps
        self.streams = streams
        self.capacity = _capacity()
        self._plan = None
        set_b12x_preparation_provider(self, self)

    def get_b12x_preparation_units(self, layer, workload):
        from b12x.norm.hyperconnection._impl import run_grouped_rmsnorm_impl

        if self._plan is None:
            self._plan = hyperconnection.plan(
                hyperconnection.Caps(
                    device=self.weight.device,
                    max_tokens=self.capacity,
                    hidden_size=self.weight.numel() // self.streams,
                    streams=self.streams,
                    lowrank=1,
                ),
                invocation={
                    "operation": "grouped_rmsnorm",
                    "zero_centered": False,
                    "eps": self.variance_epsilon,
                    "weight_dtype": str(self.weight.dtype).removeprefix("torch."),
                },
            )

        def prepare(state):
            source = torch.ones(
                (1, self.weight.numel()),
                dtype=torch.bfloat16,
                device=self.weight.device,
            )
            out = torch.empty_like(source)
            return PreparedCall(
                run=lambda: run_grouped_rmsnorm_impl(
                    source,
                    self.weight,
                    eps=self.variance_epsilon,
                    plan=state,
                    out=out,
                    zero_centered=False,
                ),
                owners=(self.weight,),
            )

        return (
            B12xPreparationUnit(
                name="V41RMSNorm",
                key=(id(self), self.capacity),
                stage="weights",
                requests=(
                    self._plan.request(
                        name=f"deepseek_v41.rmsnorm.{id(self):x}",
                        prepare_call=prepare,
                    ),
                ),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._plan is None:
            raise PreparationResourceUnavailableError("V4.1 RMSNorm has no declared plan")
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        out = torch.empty_like(x)
        _rmsnorm(
            x,
            self.weight,
            out,
            self.variance_epsilon,
            self._plan.handle,
        )
        return out.view(shape)


@torch.library.custom_op("aphrodite::dsv41_mhc_pre", mutates_args=("residual_out", "y", "post", "comb", "pre_out"))
def _mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm: torch.Tensor,
    pre: torch.Tensor,
    residual_out: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre_out: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    iterations: int,
    layer_name: LayerNameType,
    previous_output: torch.Tensor | None = None,
    previous_post: torch.Tensor | None = None,
    previous_comb: torch.Tensor | None = None,
) -> None:
    mhc_module = b12x_layer(_resolve_layer_name(layer_name))._b12x_mhc
    operation = "pre" if previous_output is None else "post_pre"
    plan = mhc_module._plan_for(operation, int(residual.shape[0]))
    scratch = current_workspace_manager().get_simultaneous(*((spec.shape, spec.dtype) for spec in plan.scratch_specs()))
    binding = mhc.bind(
        plan,
        scratch=scratch,
        tokens=int(residual.shape[0]),
        out=residual_out,
        y=y,
        post=post,
        comb=comb,
        pre_out=pre_out,
    )
    # Caller-owned outputs follow PyTorch's graph-pool lifetimes. Retaining
    # them through the binding prevents reuse across layers and graph shapes.
    retain_cuda_graph_capture_resource(scratch)
    if previous_output is None:
        mhc.run_pre(
            residual,
            fn,
            scale,
            base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=iterations,
            norm_weight=norm,
            norm_eps=rms_eps,
            pre_mix=pre,
            binding=binding,
        )
    else:
        if previous_post is None or previous_comb is None:
            raise ValueError("lagged mHC post-pre requires both previous mix tensors")
        mhc.run_post_pre(
            previous_output,
            residual,
            previous_post,
            previous_comb,
            fn,
            scale,
            base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=iterations,
            norm_weight=norm,
            norm_eps=rms_eps,
            pre_mix=pre,
            binding=binding,
        )


@_mhc_pre.register_fake
def _mhc_pre_fake(
    residual,
    fn,
    scale,
    base,
    norm,
    pre,
    residual_out,
    y,
    post,
    comb,
    pre_out,
    rms_eps,
    hc_eps,
    iterations,
    layer_name,
    previous_output=None,
    previous_post=None,
    previous_comb=None,
):
    return None


@torch.library.custom_op("aphrodite::dsv41_mhc_post", mutates_args=("out",))
def _mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    mhc_module = b12x_layer(_resolve_layer_name(layer_name))._b12x_mhc
    plan = mhc_module._plan_for("post", int(residual.shape[0]))
    mhc.run_post(x, residual, post, comb, plan=plan, out=out)


@_mhc_post.register_fake
def _mhc_post_fake(x, residual, post, comb, out, layer_name):
    return None


class B12xMHC(nn.Module):
    """Prepared V4.1 lagged mHC plans owned by their loaded decoder."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rms_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.iterations = config.hc_sinkhorn_iters
        self._plans: dict[tuple[str, int], Plan] = {}
        self._plan_capacities: tuple[int, ...] = ()
        self._initial_pre_mix: torch.Tensor | None
        self.register_buffer("_initial_pre_mix", None, persistent=False)
        self._layer_name: LayerNameType | None = None
        self._collapse_plans = {}
        if config.hc_mult != 4:
            raise ValueError("V4.1 mHC requires four streams")

    def bind_layer_name(self, name: str) -> None:
        """Record the encoded name of the decoder layer that owns this module."""
        self._layer_name = _encode_layer_name(name)

    def _plan_for(self, operation, tokens):
        tokens = int(tokens)
        index = bisect_left(self._plan_capacities, tokens)
        if tokens >= 0 and index < len(self._plan_capacities):
            return self._plans[(operation, self._plan_capacities[index])]
        capacity = self._plan_capacities[-1] if self._plan_capacities else 0
        raise PreparationResourceUnavailableError(
            f"B12x mHC {operation} live M={tokens} exceeds prepared capacity {capacity}"
        )

    def get_b12x_preparation_units(self, layer, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        if any(
            t.is_meta
            for t in (
                layer.hc_attn_fn,
                layer.hc_ffn_fn,
                layer.hc_attn_scale,
                layer.hc_ffn_scale,
                layer.hc_attn_base,
                layer.hc_ffn_base,
                layer.attn_norm.weight,
                layer.ffn_norm.weight,
            )
        ):
            return ()
        key = tuple(sorted({workload.max_tokens, *workload.fixed_token_counts}))
        self._plan_capacities = tuple(sorted({*self._plan_capacities, *key}))
        capacity = self._plan_capacities[-1]
        if self._initial_pre_mix is None or self._initial_pre_mix.shape[0] < capacity:
            self._initial_pre_mix = torch.zeros((capacity, 4), dtype=torch.float32, device=layer.hc_attn_fn.device)
            self._initial_pre_mix[:, 0].fill_(1)
        common = FrozenMapping(
            {
                "lagged_mix": True,
                "has_norm_weight": True,
                "rms_eps": self.rms_eps,
                "hc_eps": self.hc_eps,
                "sinkhorn_iters": self.iterations,
                "norm_eps": self.rms_eps,
            }
        )
        for tokens in key:
            for operation in ("pre", "post_pre"):
                if (operation, tokens) in self._plans:
                    continue
                invocation = FrozenMapping(
                    {
                        **common.to_dict(),
                        "operation": operation,
                        "has_fn_bf16": False,
                        "expanded_residual": operation == "pre"
                        and int(
                            (
                                layer.hc_attn_fn_broadcast
                                if layer.hc_attn_fn_broadcast is not None
                                else layer.hc_attn_fn
                            ).shape[1]
                        )
                        == 4 * self.hidden_size,
                    }
                )
                self._plans[(operation, tokens)] = mhc.plan(
                    _mhc_caps(layer.hc_attn_fn.device, tokens, self.hidden_size),
                    invocation=invocation,
                )
        for tokens in key:
            if ("post", tokens) not in self._plans:
                self._plans[("post", tokens)] = mhc.plan(
                    _mhc_caps(layer.hc_attn_fn.device, tokens, self.hidden_size),
                    invocation=FrozenMapping({"operation": "post", "output_mode": "provided"}),
                )
        requests = tuple(
            self._plans[(operation, tokens)].request(
                name=f"deepseek_v41.mhc.{id(layer):x}.{operation}.m{tokens}",
                prepare_call=self._prepare_call(layer, operation, tokens),
                benchmark_call=self._prepare_call(layer, operation, tokens),
            )
            for tokens in key
            for operation in ("pre", "post_pre", "post")
        )
        collapse_requests = []
        for weighted in (False, True):
            if weighted not in self._collapse_plans:
                self._collapse_plans[weighted] = mhc.plan(
                    _mhc_caps(layer.hc_attn_fn.device, workload.max_tokens, self.hidden_size),
                    invocation=FrozenMapping(
                        {
                            "operation": "collapse",
                            "output_mode": "provided",
                            "collapse_weighted": weighted,
                        }
                    ),
                )

            def prepare_collapse(state, weighted=weighted):
                from b12x.norm.mhc._impl import _run_collapse_impl

                device = layer.hc_attn_fn.device
                source = torch.randn((1, 4, self.hidden_size), dtype=torch.bfloat16, device=device)
                mix = torch.ones((1, 4), dtype=torch.float32, device=device) if weighted else None
                output = torch.empty((1, self.hidden_size), dtype=torch.bfloat16, device=device)
                return PreparedCall(
                    run=lambda: _run_collapse_impl(
                        source,
                        mix,
                        out=output,
                        _state=state,
                    )
                )

            collapse_requests.append(
                self._collapse_plans[weighted].request(
                    name=f"deepseek_v41.mhc.{id(layer):x}.collapse.weighted{int(weighted)}",
                    prepare_call=prepare_collapse,
                )
            )
        return (
            B12xPreparationUnit(
                name="V41MHC",
                key=(id(layer), self.hidden_size, key),
                requests=(*requests, *collapse_requests),
                stage="weights",
            ),
        )

    def _prepare_call(self, layer, operation, tokens):
        def prepare(state):
            from b12x.norm.mhc import _impl

            device = layer.hc_attn_fn.device
            shape = (
                (tokens, 4, self.hidden_size)
                if operation != "pre" or state.query.expanded_residual
                else (tokens, self.hidden_size)
            )
            residual = torch.empty(shape, dtype=torch.bfloat16, device=device)
            x = torch.empty((tokens, self.hidden_size), dtype=torch.bfloat16, device=device)
            post = torch.empty((tokens, 4), dtype=torch.float32, device=device)
            comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)
            pre_mix = torch.empty((tokens, 4), dtype=torch.float32, device=device)
            pre_out = torch.empty_like(pre_mix)
            out = torch.empty((tokens, 4, self.hidden_size), dtype=torch.bfloat16, device=device)
            y = torch.empty_like(x)
            next_post, next_comb = torch.empty_like(post), torch.empty_like(comb)
            scratch = [torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in state.scratch_specs()]
            binding = (
                None
                if operation == "post"
                else state.bind(
                    scratch=scratch,
                    tokens=tokens,
                    out=out,
                    y=y,
                    post=next_post,
                    comb=next_comb,
                    pre_out=pre_out,
                )
            )

            def produce():
                residual.normal_()
                x.normal_()
                post.normal_()
                comb.normal_()
                pre_mix.zero_()
                pre_mix[:, 0].fill_(1)

            if operation == "pre":
                fn = layer.hc_attn_fn_broadcast
                if fn is None:
                    fn = layer.hc_attn_fn
                run = lambda: _impl._b12x_mhc_pre_impl(
                    residual,
                    fn,
                    layer.hc_attn_scale,
                    layer.hc_attn_base,
                    rms_eps=self.rms_eps,
                    hc_eps=self.hc_eps,
                    sinkhorn_iters=self.iterations,
                    norm_weight=layer.attn_norm.weight,
                    norm_eps=self.rms_eps,
                    pre_mix=pre_mix,
                    binding=binding,
                    _state=state,
                )
            elif operation == "post_pre":
                run = lambda: _impl._b12x_mhc_post_pre_impl(
                    x,
                    residual,
                    post,
                    comb,
                    layer.hc_ffn_fn,
                    layer.hc_ffn_scale,
                    layer.hc_ffn_base,
                    rms_eps=self.rms_eps,
                    hc_eps=self.hc_eps,
                    sinkhorn_iters=self.iterations,
                    norm_weight=layer.ffn_norm.weight,
                    norm_eps=self.rms_eps,
                    pre_mix=pre_mix,
                    binding=binding,
                    _state=state,
                )
            else:
                run = lambda: _impl._b12x_mhc_post_impl(x, residual, post, comb, out=out, _state=state)
            return PreparedCall(
                run=run,
                produce=produce,
                owners=(
                    layer.hc_attn_fn,
                    layer.hc_ffn_fn,
                    layer.hc_attn_scale,
                    layer.hc_ffn_scale,
                    layer.hc_attn_base,
                    layer.hc_ffn_base,
                    layer.attn_norm.weight,
                    layer.ffn_norm.weight,
                ),
            )

        return prepare

    def pre(
        self,
        residual,
        fn,
        scale,
        base,
        norm,
        pre,
        *,
        previous_output=None,
        previous_post=None,
        previous_comb=None,
    ):
        tokens = int(residual.shape[0])
        if pre is None:
            if self._initial_pre_mix is None or tokens > self._initial_pre_mix.shape[0]:
                raise PreparationResourceUnavailableError(
                    "V4.1 mHC has no prepared initial coefficients for this capacity"
                )
            pre = self._initial_pre_mix[:tokens]
        residual_out = torch.empty((tokens, 4, self.hidden_size), dtype=residual.dtype, device=residual.device)
        y = torch.empty((tokens, self.hidden_size), dtype=residual.dtype, device=residual.device)
        post = torch.empty((tokens, 4), dtype=torch.float32, device=residual.device)
        comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=residual.device)
        pre_out = torch.empty((tokens, 4), dtype=torch.float32, device=residual.device)
        _mhc_pre(
            residual,
            fn,
            scale,
            base,
            norm,
            pre,
            residual_out,
            y,
            post,
            comb,
            pre_out,
            self.rms_eps,
            self.hc_eps,
            self.iterations,
            self._layer_name,
            previous_output,
            previous_post,
            previous_comb,
        )
        return residual_out, post, comb, y, pre_out

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(
            residual,
            fn,
            scale,
            base,
            norm,
            pre,
            previous_output=x,
            previous_post=post,
            previous_comb=comb,
        )

    def post(self, x, residual, post, comb):
        out = torch.empty_like(residual)
        _mhc_post(x, residual, post, comb, out, self._layer_name)
        return out

    def collapse(self, state: torch.Tensor, pre: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.empty(
            (state.shape[0], state.shape[-1]),
            dtype=state.dtype,
            device=state.device,
        )
        return mhc.run_collapse(
            state,
            pre,
            out=out,
            plan=self._collapse_plans[pre is not None],
        )
