# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native ratio-one/two compression with request-owned partial-state rings."""

import torch
import torch.nn as nn
from b12x.attention import mla_compress
from b12x.gemm import bf16_gemv
from b12x.preparation import PreparedCall

from aphrodite.model_executor.layers.attention_layer_base import AttentionLayerBase
from aphrodite.model_executor.layers.linear import MergedColumnParallelLinear
from aphrodite.triton_utils import tl, triton
from aphrodite.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    set_b12x_preparation_provider,
)
from aphrodite.v1.attention.backend import MultipleOf
from aphrodite.v1.kv_cache_interface import CircularBufferSpec

from .sparse_mla import DeepseekV41B12xBackend


@triton.jit(do_not_specialize=["stride", "table_stride"])
def _load_partial_state(
    Pool,
    Table,
    Starts,
    Positions,
    Values,
    Gates,
    Tags,
    LocalIds,
    Counts,
    stride,
    table_stride,
    PAGE: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 512)
    nr = tl.load(Counts + 1)
    nt = tl.load(Counts)
    first = tl.load(Starts + r)
    end = tl.load(Starts + r + 1)
    pos = tl.load(Positions + r)
    live = (r < nr) & (first >= 0) & (first < end) & (end <= nt)
    live = live & (pos >= 0)
    predecessor = pos - 1
    needed = live & (pos > 0) & (pos % 2 == 1)
    page = tl.load(Table + r * table_stride, needed, other=0).to(tl.int64)
    valid = needed & (page > 0)
    base = Pool + page * stride + (predecessor % PAGE) * 1024
    value = tl.load(base + col, valid, other=0)
    gate = tl.load(base + 512 + col, valid, other=0)
    tl.store(Values + r * 512 + col, value)
    tl.store(Gates + r * 512 + col, gate)
    tl.store(Tags + r, tl.where(valid, predecessor, -1))
    tl.store(LocalIds + r, tl.where(live, r, -1))


@triton.jit(do_not_specialize=["stride"])
def _save_partial_states(Pool, Slots, Values, Gates, Counts, stride, PAGE: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 512)
    nt = tl.load(Counts)
    slot = tl.load(Slots + token).to(tl.int64)
    valid = (token < nt) & (slot >= PAGE)
    base = Pool + (slot // PAGE) * stride + (slot % PAGE) * 1024
    value = tl.load(Values + token * 512 + col, valid, other=0)
    gate = tl.load(Gates + token * 512 + col, valid, other=0)
    tl.store(base + col, value, valid)
    tl.store(base + 512 + col, gate, valid)


class CompressorBackend(DeepseekV41B12xBackend):
    @staticmethod
    def get_name():
        return "B12X_DSV41_COMPRESSOR"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(1)]

    @classmethod
    def is_ssm(cls):
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls):
        return True


class CompressorStateCache(nn.Module, AttentionLayerBase):
    def __init__(self, config, prefix):
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        spec = config.speculative_config
        draft_tokens = spec.num_speculative_tokens if spec is not None else 0
        # Keep draft, bonus, and previous partial states until rejection selects the boundary.
        self.block_size = max(8, 1 << (draft_tokens + 1).bit_length())
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, cache):
        self.kv_cache = cache.view(cache.shape[0], -1)

    def get_kv_cache_spec(self, config):
        return CircularBufferSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=1024,
            head_size_v=0,
            dtype=torch.float32,
            dcp_sharded=False,
            max_tp_shards=1,
        )

    def get_attn_backend(self):
        return CompressorBackend

    def forward(self):
        raise RuntimeError("compression state is consumed by its owning compressor")


class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        aphrodite_config,
        compress_ratio,
        hidden_size,
        head_dim,
        prefix="",
    ):
        super().__init__()
        if compress_ratio not in (1, 2) or head_dim != 512:
            raise ValueError("V4.1 compression requires ratio1/2 and head512")
        self.prefix, self.compress_ratio = prefix, compress_ratio
        self.capacity = aphrodite_config.scheduler_config.max_num_batched_tokens
        self.requests = aphrodite_config.scheduler_config.max_num_seqs
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [512] * compress_ratio,
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
            params_dtype=torch.bfloat16,
        )
        self.norm = nn.Module()
        self.norm.register_parameter("weight", nn.Parameter(torch.ones(512, dtype=torch.float32), requires_grad=False))
        self.state_cache = (
            CompressorStateCache(aphrodite_config, f"{prefix}.state_cache") if compress_ratio == 2 else None
        )
        self._plan = None
        self._projection_plans = {}
        self._values = self._latent = self._emitted = self._slots = None
        self._gates = self._pending_values = self._pending_gates = None
        self._pending_tags = self._local_ids = self._row_ids = None
        self._primer_starts = self._primer_positions = self._primer_state_ids = None
        self._primer_slots = self._primer_counts = None
        set_b12x_preparation_provider(self, self)

    def _ensure_capacity_buffers(self, caps) -> None:
        """Allocate reusable serving/priming storage only during preparation."""
        if self._values is not None:
            if self._values.shape[0] != caps.max_tokens:
                raise PreparationResourceUnavailableError("compressor capacity changed after preparation")
            return
        device, t, r = caps.device, caps.max_tokens, caps.max_requests

        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        self._values = alloc((t, 512), torch.float32 if caps.ratio == 2 else torch.bfloat16)
        self._latent = alloc((t, 512), torch.bfloat16)
        self._emitted = alloc((t,), torch.bool)
        self._slots = alloc((t,), torch.int64)
        self._row_ids = torch.arange(r, dtype=torch.int64, device=device)
        self._primer_starts = torch.empty((r + 1,), dtype=torch.int32, device=device)
        self._primer_positions = torch.empty((r,), dtype=torch.int64, device=device)
        self._primer_state_ids = torch.empty((r,), dtype=torch.int64, device=device)
        self._primer_slots = torch.empty((t,), dtype=torch.int64, device=device)
        self._primer_counts = torch.empty((2,), dtype=torch.int32, device=device)
        if caps.ratio == 2:
            self._gates = alloc((t, 512), torch.float32)
            self._pending_values = alloc((r, 512), torch.float32)
            self._pending_gates = alloc((r, 512), torch.float32)
            self._pending_tags = alloc((r,), torch.int64)
            self._local_ids = alloc((r,), torch.int64)

    def _reset_primer_state(self) -> None:
        """Restore all reusable mutable capacity storage after a trial."""
        self._values.zero_()
        self._latent.zero_()
        self._emitted.zero_()
        self._slots.fill_(-1)
        self._primer_starts.zero_()
        self._primer_positions.fill_(-1)
        self._primer_state_ids.fill_(-1)
        self._primer_slots.fill_(-1)
        self._primer_counts.zero_()
        if self.compress_ratio == 2:
            self._gates.zero_()
            self._pending_values.zero_()
            self._pending_gates.zero_()
            self._pending_tags.fill_(-1)
            self._local_ids.fill_(-1)

    def _produce_primer_input(self) -> None:
        """Prime native code with a real nonzero ratio-one/pair invocation."""
        self._reset_primer_state()
        tokens = 2 if self.compress_ratio == 2 else 1
        self._values[0].fill_(0.25)
        if tokens == 2:
            self._values[1].fill_(1.5)
            self._gates[0].fill_(-0.75)
            self._gates[1].fill_(0.5)
        self._primer_starts[1].fill_(tokens)
        self._primer_positions[0].zero_()
        self._primer_state_ids[0].zero_()
        self._primer_slots[0].fill_(5)
        if tokens == 2:
            self._primer_slots[1].fill_(6)
        self._primer_counts[0].fill_(tokens)
        self._primer_counts[1].fill_(1)

    def _primer_binding(self, state):
        return state.bind(
            values=self._values,
            gates=self._gates,
            weight=self.norm.weight,
            query_start_loc=self._primer_starts,
            positions=self._primer_positions,
            state_ids=self._primer_state_ids,
            destination_slots=self._primer_slots,
            live_counts=self._primer_counts,
            out=self._latent,
            emitted=self._emitted,
            emitted_slots=self._slots,
            pending_values=self._pending_values,
            pending_gates=self._pending_gates,
            pending_position=self._pending_tags,
        )

    def _projection_unit(self, workload):
        weight = self.fused_wkv_wgate.weight
        capacities = tuple(sorted({workload.max_tokens, *workload.token_counts}))
        self._projection_capacity = workload.max_tokens
        dtype = torch.float32 if self.compress_ratio == 2 else torch.bfloat16

        def prepare(state):
            source = torch.empty(
                (state.query.max_rows, weight.shape[1]),
                dtype=torch.bfloat16,
                device=weight.device,
            )
            outputs = tuple(
                torch.empty(
                    (state.query.max_rows, 512),
                    dtype=dtype,
                    device=weight.device,
                )
                for _ in range(self.compress_ratio)
            )

            def run():
                for index, output in enumerate(outputs):
                    state.run(source, weight[index * 512 : (index + 1) * 512], out=output)
                return outputs

            return PreparedCall(
                run=run,
                output=outputs,
                produce=lambda: source.normal_(std=0.25),
                owners=(weight,),
            )

        requests = []
        for rows in capacities:
            if rows not in self._projection_plans:
                self._projection_plans[rows] = bf16_gemv.plan(
                    bf16_gemv.GemvQuery(
                        source_dtype="bfloat16",
                        weight_dtype="bfloat16",
                        output_dtype=str(dtype).removeprefix("torch."),
                        max_rows=rows,
                        in_features=weight.shape[1],
                        out_features=512,
                        source_contiguous=True,
                        source_aligned=True,
                        weight_contiguous=weight.is_contiguous(),
                        weight_aligned=weight.data_ptr() % 16 == 0,
                    )
                )
            requests.append(
                self._projection_plans[rows].request(
                    name=f"{self.prefix}/projection.m{rows}",
                    prepare_call=prepare,
                    benchmark_call=prepare,
                )
            )
        return B12xPreparationUnit(
            name="V41CompressorProjection",
            key=(self.prefix, capacities),
            requests=tuple(requests),
            stage="weights",
        )

    def _project(self, hidden_states):
        rows = hidden_states.shape[0]
        plan = self._projection_plans.get(rows)
        if plan is None:
            plan = self._projection_plans[self._projection_capacity]
        for index, output in enumerate((self._values, self._gates)[: self.compress_ratio]):
            bf16_gemv.mm(
                hidden_states,
                self.fused_wkv_wgate.weight[index * 512 : (index + 1) * 512],
                out=output[:rows],
                output_dtype=output.dtype,
                plan=plan,
            )

    def get_b12x_preparation_units(self, layer: object, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        if workload.max_tokens > self.capacity:
            return ()
        if self.fused_wkv_wgate.weight.is_meta or self.norm.weight.is_meta:
            return ()
        if self._plan is None:
            device = self.fused_wkv_wgate.weight.device
            self._plan = mla_compress.plan(
                mla_compress.Caps(
                    device=device,
                    max_tokens=self.capacity,
                    max_requests=self.requests,
                    max_states=self.requests,
                    ratio=self.compress_ratio,
                )
            )
        declaration = self._plan

        def make_call(state):
            self._ensure_capacity_buffers(state.caps)
            binding = self._primer_binding(state)
            return PreparedCall(
                run=lambda: state.run(binding),
                produce=self._produce_primer_input,
                reset=self._reset_primer_state,
                restore=self._reset_primer_state,
                owners=(
                    self.norm.weight,
                    self._values,
                    self._gates,
                    self._primer_starts,
                    self._primer_positions,
                    self._primer_state_ids,
                    self._primer_slots,
                    self._primer_counts,
                    self._latent,
                    self._emitted,
                    self._slots,
                    self._pending_values,
                    self._pending_gates,
                    self._pending_tags,
                    self._local_ids,
                ),
            )

        request = declaration.request(
            name=f"{self.prefix}/mla-compress",
            prepare_call=make_call,
            benchmark_call=make_call,
        )
        return (
            self._projection_unit(workload),
            B12xPreparationUnit(
                name="DeepseekCompressor",
                key=(self.prefix, self.compress_ratio, self.capacity, self.requests),
                requests=(request,),
                stage="state",
            ),
        )

    def forward(self, hidden_states, metadata, state_metadata=None):
        if self._plan is None or self._plan.prepared is None:
            raise PreparationResourceUnavailableError("compressor plan must be prepared before forwarding")
        rows = hidden_states.shape[0]
        self._project(hidden_states)
        if self.compress_ratio == 2:
            if state_metadata is None:
                raise RuntimeError("ratio2 compression requires request-state metadata")
            pool = self.state_cache.kv_cache
            _load_partial_state[(self.requests,)](
                pool,
                state_metadata.block_table,
                state_metadata.query_start_loc,
                state_metadata.request_positions,
                self._pending_values,
                self._pending_gates,
                self._pending_tags,
                self._local_ids,
                state_metadata.live_counts,
                pool.stride(0),
                state_metadata.block_table.stride(0),
                state_metadata.block_size,
            )
            _save_partial_states[(self.capacity,)](
                pool,
                state_metadata.slot_mapping,
                self._values,
                self._gates,
                state_metadata.live_counts,
                pool.stride(0),
                state_metadata.block_size,
            )
            ids = self._local_ids
        else:
            ids = self._row_ids
        binding = mla_compress.bind(
            self._plan,
            values=self._values,
            gates=self._gates,
            weight=self.norm.weight,
            query_start_loc=metadata.query_start_loc,
            positions=metadata.request_positions,
            state_ids=ids,
            destination_slots=metadata.slot_mapping,
            live_counts=metadata.live_counts,
            out=self._latent,
            emitted=self._emitted,
            emitted_slots=self._slots,
            pending_values=self._pending_values,
            pending_gates=self._pending_gates,
            pending_position=self._pending_tags,
        )
        mla_compress.run(binding)
        return self._latent[:rows], self._slots[:rows]
