# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare b12x plans held by the loaded model at two lifecycle points.

The ``weights`` stage selects, compiles and primes weight-owned plans before
memory profiling. The ``state`` stage binds plans to the final KV/state pools
before warmup or graph capture. Disposable profiling pools prepare defaults
and release only their own plans after their graphs have been destroyed.
Both stages collect one unit per
layer from ``layer.b12x_preparation_provider.get_b12x_preparation_units`` and
hand the units' requests to one ``PreparationSession`` per worker.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

import torch

from aphrodite.platforms import current_platform
from aphrodite.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    b12x_preparation_token_counts,
    b12x_unit_providers,
    has_b12x,
    scope_b12x_unit_calls,
)

if TYPE_CHECKING:
    from b12x.preparation import PreparationRequest

    from aphrodite.v1.worker.gpu_worker import Worker


def b12x_native_supported(worker: Worker) -> bool:
    """Opt in by the loaded model contract, never by installed package alone."""
    return bool(getattr(worker.get_model(), "supports_b12x_native_preparation", False))


def _validate_native_worker(worker: Worker) -> None:
    if not has_b12x():
        raise RuntimeError("Native SM120 preparation requires FlashInfer's bundled B12x provider")
    if not current_platform.is_cuda() or not current_platform.is_device_capability_family(120):
        raise RuntimeError("Native B12x preparation requires CUDA device family 120")
    if not worker.use_v2_model_runner:
        raise RuntimeError("Native B12x preparation requires the V2 model runner")
    parallel = worker.aphrodite_config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1 or parallel.enable_dbo:
        raise NotImplementedError("Native SM120 preparation does not support DCP, PCP or DBO")


def _draft_lane(worker: Worker) -> int:
    runner = worker.model_runner
    return int(getattr(runner, "_draft_workspace_lane", 0) or 0)


def _planned_decode_counts(
    worker: Worker, *, capture_sizes: tuple[int, ...], speculative_tokens: int
) -> tuple[int, ...]:
    """Token counts of the decode graphs the runner will capture.

    Both stages must declare the same counts: a plan prepared in the weights
    stage serves every graph, and a count that first appears in the state
    stage would reach a weight-only family after its winners were chosen. The
    graph manager knows the exact set once the KV cache exists; before that
    uniform and variable-length decode counts follow the same configuration.
    """
    runner = worker.model_runner
    manager = getattr(runner, "cudagraph_manager", None)
    planned_counts = getattr(manager, "planned_token_counts", None)
    if callable(planned_counts):
        return tuple(int(count) for count in planned_counts() if int(count) > 0)
    query_len = int(getattr(runner, "decode_query_len", 0) or 0) or 1 + int(speculative_tokens)
    if query_len <= 1 or not capture_sizes:
        return ()
    compilation = worker.aphrodite_config.compilation_config
    limit = int(getattr(compilation, "max_cudagraph_capture_size", None) or max(capture_sizes))
    max_reqs = int(worker.scheduler_config.max_num_seqs)
    counts = set()
    for size in capture_sizes:
        rounded = -(-int(size) // query_len) * query_len
        if rounded // query_len <= max_reqs and rounded <= limit:
            counts.add(rounded)
    return tuple(sorted(counts))


def b12x_workload(worker: Worker, *, stage: str, lane: int = 0) -> B12xWorkload:
    """Describe the serving shapes of one stage without constructing plans."""
    compilation = worker.aphrodite_config.compilation_config
    capture_sizes = tuple(compilation.cudagraph_capture_sizes or ())
    compile_sizes = tuple(size for size in (compilation.compile_sizes or ()) if isinstance(size, int))
    endpoints = tuple(compile_range.end for compile_range in compilation.get_compile_ranges())
    spec = worker.aphrodite_config.speculative_config
    speculative_tokens = 0 if spec is None else int(getattr(spec, "num_speculative_tokens", 0))
    max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
    planned = _planned_decode_counts(
        worker,
        capture_sizes=capture_sizes,
        speculative_tokens=speculative_tokens,
    )
    token_counts = b12x_preparation_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=(*capture_sizes, *planned),
        compile_sizes=compile_sizes,
        compile_range_endpoints=endpoints,
        speculative_tokens=speculative_tokens,
    )
    # Sonar's scheduler warmup can cap its request count by available blocks.
    # Declare every reachable prefill row count before either cache pool exists.
    prompt_len = int(worker.model_runner.decode_query_len) + 1
    warmup_reqs = min(int(worker.scheduler_config.max_num_seqs), max_tokens // prompt_len)
    token_counts = tuple(
        sorted(
            {
                *token_counts,
                *(prompt_len * reqs for reqs in range(1, warmup_reqs + 1)),
            }
        )
    )
    fixed = {int(count) for count in (*capture_sizes, *compile_sizes, *planned)}
    fixed_token_counts = tuple(count for count in token_counts if count in fixed and count < max_tokens)
    dtype = worker.model_config.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        dtype = torch.bfloat16
    return B12xWorkload(
        stage=cast(Literal["weights", "state"], stage),
        token_counts=token_counts,
        fixed_token_counts=fixed_token_counts,
        output_dtype=dtype,
        max_tokens=max_tokens,
        max_seqs=int(worker.scheduler_config.max_num_seqs),
        lane=lane,
    )


def _units_from_modules(
    model: torch.nn.Module,
    workload: B12xWorkload,
    *,
    seen: set[int] | None = None,
    module_lanes: dict[int, tuple[int, ...]] | None = None,
) -> Iterable[B12xPreparationUnit]:
    if seen is None:
        seen = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        provider = getattr(module, "b12x_preparation_provider", None)
        hook = getattr(provider, "get_b12x_preparation_units", None)
        if not callable(hook):
            continue
        for unit in hook(module, workload):
            if not isinstance(unit, B12xPreparationUnit):
                raise TypeError(f"{type(provider).__qualname__} returned a non-unit preparation value")
            lanes = (workload.lane,) if module_lanes is None else module_lanes.get(id(module), (workload.lane,))
            yield replace(unit, workspace_lanes=lanes)


def _draft_workload(worker: Worker, workload: B12xWorkload, *, lane: int):
    """Include the parallel draft's query and sequential sampling row counts."""
    speculator = getattr(worker.model_runner, "speculator", None)
    query_len = int(getattr(speculator, "num_query_per_req", 0) or 0)
    if not query_len:
        return replace(workload, lane=lane)
    max_reqs = min(workload.max_seqs, workload.max_tokens // query_len)
    counts = set(workload.token_counts)
    draft_counts = set(range(1, max_reqs + 1))
    draft_counts.update(reqs * query_len for reqs in range(1, max_reqs + 1))
    manager = getattr(speculator, "query_cudagraph_manager", None)
    planned = getattr(manager, "planned_token_counts", None)
    if callable(planned):
        draft_counts.update(int(count) for count in planned() if int(count) > 0)
    counts.update(draft_counts)
    fixed = set(workload.fixed_token_counts)
    fixed.update(count for count in draft_counts if count < workload.max_tokens)
    return replace(
        workload,
        token_counts=tuple(sorted(counts)),
        fixed_token_counts=tuple(sorted(fixed)),
        lane=lane,
    )


def collect_b12x_units(worker: Worker, workload: B12xWorkload) -> list[B12xPreparationUnit]:
    """Collect every unit of one stage from the model, the draft, and comms."""
    units: list[B12xPreparationUnit] = []
    model = worker.get_model()
    draft = worker.get_draft_model()
    lane = _draft_lane(worker) if draft is not None else 0
    lanes_by_module: dict[int, set[int]] = {}
    for owner, owner_lane in ((model, 0), (draft, lane)):
        if owner is not None:
            for module in owner.modules():
                lanes_by_module.setdefault(id(module), set()).add(owner_lane)
    module_lanes = {key: tuple(sorted(lanes)) for key, lanes in lanes_by_module.items()}
    if draft is not None:
        workload = _draft_workload(worker, workload, lane=0)
    # Target and draft can share the same embedding and output-head modules.
    seen: set[int] = set()
    units.extend(_units_from_modules(model, workload, seen=seen, module_lanes=module_lanes))
    if draft is not None:
        draft_workload = replace(workload, lane=lane)
        from aphrodite.v1.worker.workspace import use_workspace_lane

        with use_workspace_lane(lane):
            draft_units = list(_units_from_modules(draft, draft_workload, seen=seen, module_lanes=module_lanes))
        if lane:
            draft_units = [scope_b12x_unit_calls(unit, lane) for unit in draft_units]
        units.extend(draft_units)
    for provider in b12x_unit_providers():
        hook = getattr(provider, "get_b12x_preparation_units", None)
        if callable(hook):
            owner_ref = getattr(provider, "b12x_preparation_owner", None)
            owner = None if owner_ref is None else owner_ref()
            if owner_ref is not None and id(owner) not in module_lanes:
                continue
            lanes = tuple(sorted({0, lane})) if owner is None else module_lanes[id(owner)]
            owner_lane = lanes[0]
            from aphrodite.v1.worker.workspace import use_workspace_lane

            with use_workspace_lane(owner_lane):
                provider_units = hook(provider, replace(workload, lane=owner_lane))
                for unit in provider_units:
                    if not isinstance(unit, B12xPreparationUnit):
                        raise TypeError("B12x provider returned a non-unit value")
                    if len(lanes) == 1 and owner_lane:
                        unit = scope_b12x_unit_calls(unit, owner_lane)
                    units.append(replace(unit, workspace_lanes=lanes))
    # The state stage also carries the weights-stage units: their prepared
    # plans are skipped by the session, and any plan a layer declared for a
    # count that only the state stage exposes is prepared before capture.
    own = [unit for unit in units if unit.stage == workload.stage]
    if workload.stage == "state":
        units = own + [unit for unit in units if unit.stage != workload.stage]
    else:
        units = own
    names = Counter(request.name for unit in units for request in unit.requests)
    duplicates = sorted(name for name, count in names.items() if count > 1)
    if duplicates:
        raise ValueError(f"conflicting b12x preparation request names: {duplicates}")
    return units


def b12x_batches(units: Iterable[B12xPreparationUnit], *, autotune: bool = True):
    """Group requests: timed selection first, then default-only preparation.

    With autotune disabled each rank compiles and primes its own defaults
    or explicit pins in the calling process.
    """
    tuned: list[PreparationRequest] = []
    defaults: list[PreparationRequest] = []
    for unit in units:
        (tuned if unit.autotune and autotune else defaults).extend(unit.requests)
    batches = []
    if tuned:
        batches.append((tuple(tuned), True))
    if defaults:
        batches.append((tuple(defaults), False))
    return batches


def _request_workspace_lanes(units: Iterable[B12xPreparationUnit]):
    return {request.name: unit.workspace_lanes for unit in units for request in unit.requests}


def get_b12x_session(worker: Worker):
    """Return the worker's preparation session, creating it on first use."""
    session = getattr(worker, "_b12x_session", None)
    if session is not None and session.state != "CLOSED":
        return session
    from b12x.preparation import PreparationSession

    config = worker.aphrodite_config
    spec = config.speculative_config
    namespace = {
        "model": str(config.model_config.model),
        "dtype": str(config.model_config.dtype),
        "kv_cache_dtype": str(config.cache_config.cache_dtype),
        "tensor_parallel": int(config.parallel_config.tensor_parallel_size),
        "pipeline_parallel": int(config.parallel_config.pipeline_parallel_size),
        "decode_context_parallel": int(getattr(config.parallel_config, "decode_context_parallel_size", 1) or 1),
        "draft_model": None if spec is None else str(getattr(spec, "model", None)),
        "speculative_method": None if spec is None else str(getattr(spec, "method", None)),
    }
    session = PreparationSession(
        device=worker.device,
        autotune=True,
        namespace=namespace,
    )
    worker._b12x_session = session
    if session.autotune and os.environ.get("B12X_AUTOTUNE", "1") != "0":
        from aphrodite.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        ranks = tuple(sorted(int(rank) for rank in tp_group.ranks))
        session.configure_tuning_shard(int(worker.rank), ranks)
    return session


def begin_b12x_preparation(
    worker: Worker,
    *,
    stage: str,
    autotune: bool | None = None,
    initial_error: Exception | None = None,
):
    """Construct a complete-world round, including ranks whose discovery failed."""
    from aphrodite.distributed.parallel_state import get_world_group
    from aphrodite.v1.worker.b12x_startup import B12xPreparationCoordinator
    from aphrodite.v1.worker.workspace import current_workspace_manager

    units: list[B12xPreparationUnit] = []
    batches = []
    session = None
    error = None
    try:
        if initial_error is not None:
            raise initial_error
        _validate_native_worker(worker)
        workload = b12x_workload(worker, stage="state" if stage == "bind" else stage)
        units = collect_b12x_units(worker, workload)
        if stage == "weights" and not units:
            raise RuntimeError("Native model declared no B12x preparation units")
        enabled = os.environ.get("B12X_AUTOTUNE", "1") != "0"
        batches = b12x_batches(units, autotune=enabled if autotune is None else autotune)
        session = get_b12x_session(worker) if batches else None
        if session is not None:
            compile_workers = os.environ.get(f"B12X_{stage.upper()}_COMPILE_WORKERS")
            session.configure_compile_workers(None if compile_workers is None else int(compile_workers))
    except Exception as failure:
        # Failed discovery must still reach the complete-world control round.
        error = failure
        batches = []
    fresh = tuple(request.plan for requests, _ in batches for request in requests if request.plan.prepared is None)
    coordinator = B12xPreparationCoordinator(
        session,
        batches,
        global_rank=int(worker.rank),
        world_group=get_world_group(),
        workspace=current_workspace_manager() if batches else None,
        request_workspace_lanes=_request_workspace_lanes(units),
        auxiliary_stream_requests=frozenset(name for unit in units for name in unit.auxiliary_stream_requests),
        initial_error=error,
    )
    coordinator.prepared_batch = B12xPreparedBatch(session, fresh)
    return coordinator


def run_b12x_preparation(
    worker: Worker,
    *,
    stage: str,
    autotune: bool | None = None,
    initial_error: Exception | None = None,
) -> B12xPreparedBatch | None:
    """Finish preparation on every rank before the next model collective."""
    if not b12x_native_supported(worker):
        return None
    from aphrodite.config import set_current_aphrodite_config

    with set_current_aphrodite_config(worker.aphrodite_config):
        coordinator = begin_b12x_preparation(worker, stage=stage, autotune=autotune, initial_error=initial_error)
        try:
            outcome = coordinator.status()
            while not outcome["done"]:
                outcome = coordinator.advance()
            if outcome["error"] is not None:
                raise RuntimeError(f"B12x {stage} preparation failed: {outcome['error']}")
        except BaseException:
            coordinator.abort()
            if coordinator.prepared_batch is not None:
                coordinator.prepared_batch.release()
            raise
        coordinator.abort()
        return coordinator.prepared_batch


class B12xPreparedBatch:
    """Plans first prepared outside the executor rounds, released explicitly.

    Plans that were already prepared when the batch was built are not owned
    by it; releasing the batch leaves them installed.
    """

    def __init__(self, session, plans):
        self.session = session
        self.plans = tuple(plans)

    def release(self) -> None:
        plans, self.plans = self.plans, ()
        if plans:
            torch.accelerator.synchronize()
            self.session.release_many(reversed(plans))


def prepare_b12x_state(worker: Worker, kv_cache_config) -> None:
    """Tune with final cache layout before the serving pool consumes free VRAM."""
    if not b12x_native_supported(worker):
        return
    from aphrodite.config import set_current_aphrodite_config
    from aphrodite.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from aphrodite.v1.worker.gpu.cudagraph_utils import _teardown_profiling_state

    config = worker.aphrodite_config
    if config.attention_config.hisparse_config is not None:
        # HiSparse owns a separate shared host-pool lifecycle; bind its real
        # pools below instead of constructing a conflicting temporary pool.
        return
    saved_blocks = worker.cache_config.num_gpu_blocks
    batch = None
    try:
        with set_current_aphrodite_config(config):
            error = None
            try:
                builder = get_kv_cache_config_builder(config)
                temporary = builder.get_kv_cache_config_from_groups(
                    config,
                    kv_cache_config.kv_cache_groups,
                    num_blocks=min(
                        kv_cache_config.num_blocks,
                        max(32, 2 + 4 * worker.scheduler_config.max_num_seqs),
                    ),
                )
                temporary.kv_cache_layout = kv_cache_config.kv_cache_layout
                worker.model_runner.initialize_kv_cache(temporary, is_profiling=True)
                worker.cache_config.num_gpu_blocks = temporary.num_blocks
            except Exception as failure:
                error = failure
            batch = run_b12x_preparation(worker, stage="state", initial_error=error)
    finally:
        try:
            # Graphs and model cache aliases must die before their plans.
            _teardown_profiling_state(worker.model_runner)
        finally:
            if batch is not None:
                batch.release()
            worker.cache_config.num_gpu_blocks = saved_blocks
