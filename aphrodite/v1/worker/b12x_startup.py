# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local preparation with complete-world exchanges at dependency boundaries.

Each rank completes its independent candidate shards before rank zero gathers
all local winners and broadcasts the selected configurations. Fixed collective
warmups have separate readiness exchanges. Progress reporting does not stop
local work or enter model collectives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import msgspec

if TYPE_CHECKING:
    from b12x.preparation import PreparationRequest, TuningCacheRequirement, TuningRequirement

    from aphrodite.distributed.utils import StatelessProcessGroup
    from aphrodite.model_executor.warmup.b12x_prepare import B12xPreparedBatch

_TuningResult = tuple[
    str,
    tuple[int, ...],
    dict[str, object] | None,
    float | None,
    int | None,
    int,
    tuple[str, ...],
]

_CacheWire = tuple[tuple[int, ...], dict[str, object], dict[str, object]]


def _cache_to_wire(cache: TuningCacheRequirement | None) -> _CacheWire | None:
    if cache is None:
        return None
    return cache.ranks, cache.identity.to_dict(), cache.records.to_dict()


def _cache_from_wire(cache: _CacheWire | None) -> TuningCacheRequirement | None:
    if cache is None:
        return None
    from b12x.preparation import TuningCacheRequirement

    return TuningCacheRequirement(tuple(cache[0]), cache[1], cache[2])


def _scoped_key(key: str, ranks: tuple[int, ...]) -> str:
    return f"{','.join(str(rank) for rank in ranks)}|{key}"


def _unscoped_key(key: str) -> str:
    return key.split("|", 1)[1]


class B12xPreparationCoordinator:
    """Run local preparation and consolidate results at required boundaries."""

    def __init__(
        self,
        session,
        batches,
        *,
        global_rank: int,
        world_group,
        request_workspace_lanes: dict[str, tuple[int, ...]],
        auxiliary_stream_requests: frozenset[str],
        workspace=None,
        initial_error: Exception | None = None,
    ) -> None:
        if type(global_rank) is not int or global_rank < 0:
            raise ValueError("global_rank must be a nonnegative integer")
        self.session = session
        self.prepared_batch: B12xPreparedBatch | None = None
        self._workspace = workspace
        self._request_workspace_lanes = request_workspace_lanes
        self._auxiliary_stream_requests = auxiliary_stream_requests
        self._active_requests: tuple[PreparationRequest, ...] = ()
        self.global_rank = global_rank
        self.world_group = world_group
        self.world_ranks = _world_ranks(world_group)
        if global_rank not in self.world_ranks:
            raise ValueError("global rank is not in preparation control domain")

        self._round = 0
        self._control = None
        if len(self.world_ranks) > 1:
            group = _get_control_group(world_group)
            sequence = getattr(group, "_b12x_sequence", 0)
            group._b12x_sequence = sequence + 1
            import torch.distributed as dist

            self._control = dist.PrefixStore(f"stage-{sequence}", group.store)
        self._authorized_key: str | None = None
        self._authorized_tuning: tuple[TuningRequirement, ...] | None = None
        self._authorized_cache: tuple[TuningCacheRequirement, ...] | None = None
        self._stop = False
        self._error: dict[str, object] | None = None
        self._last_progress = None
        self._cleanup_complete = False
        self._closed = False

        self._batches = [(tuple(requests), bool(autotune)) for requests, autotune in batches if requests]
        self._native = bool(self._batches)
        self._job = None
        self._local_done = not self._native
        self._global_done = False
        try:
            if initial_error is not None:
                raise initial_error
            if self._native:
                if session is None:
                    raise RuntimeError("native preparation requests require a session")
                self._begin_next_batch()
        except Exception as error:
            self._fail(error)

    def _begin_next_batch(self) -> None:
        requests, autotune = self._batches.pop(0)
        self._active_requests = requests
        self._job = self.session.begin(requests, autotune=autotune)

    def status(self) -> dict[str, object]:
        """Return the current preparation outcome."""
        return self._outcome()

    def advance(self) -> dict[str, object]:
        """Advance locally; exchange metadata only when local work is blocked."""
        if self._closed:
            return self._outcome()
        if self._control is not None:
            if self._stop:
                self._control.set("stop", b"1")
            self._stop |= self._control.check(["stop"])
            if self._control.check(["failed"]):
                self._error = msgspec.msgpack.decode(self._control.get("failed"))
                self._safe_close()
        try:
            self._advance_local()
        except BaseException as error:
            self._fail(error)

        if not (self._local_done or self._error or self._ready() or self._ready_tuning() or self._ready_cache()):
            return self._outcome()

        decision = self._exchange()
        self._stop |= decision["stop"]
        if decision["error"] is not None:
            self._error = decision["error"]
            self._stop = True
            self._safe_close()
        else:
            cache = self._ready_cache()
            if cache is not None:
                self._authorized_cache = () if self._stop else decision["caches"][cache.ranks]
            authorization = decision["collective"]
            self._authorized_key = (
                authorization[0] if authorization is not None and self.global_rank in authorization[1] else None
            )
            if self._ready_tuning():
                from b12x.preparation import TuningRequirement

                self._authorized_tuning = tuple(
                    TuningRequirement(
                        _unscoped_key(key),
                        ranks,
                        assignment,
                        latency,
                        index,
                        rejected,
                        programs,
                    )
                    for (
                        key,
                        ranks,
                        assignment,
                        latency,
                        index,
                        rejected,
                        _,
                        programs,
                    ) in decision["tuning"]
                    if self.global_rank in ranks
                )
        self._global_done = decision["done"]
        if self._global_done:
            self._safe_close()
            self._closed = True
        self._round += 1
        return self._outcome()

    def _exchange(self):
        payload = self._payload()
        if self._control is None:
            return self._decision([payload])
        prefix = f"round-{self._round}"
        self._control.set(f"{prefix}/{self.global_rank}", msgspec.msgpack.encode(payload))
        if self.global_rank == self.world_ranks[0]:
            gathered = [msgspec.msgpack.decode(self._control.get(f"{prefix}/{rank}")) for rank in self.world_ranks]
            try:
                decision = self._decision(gathered)
            except Exception as error:
                self._record_error(error)
                decision = dict(
                    stop=True,
                    error=self._error,
                    collective=None,
                    tuning=(),
                    caches={},
                    done=False,
                )
            wire_decision = dict(
                decision,
                caches=[
                    (ranks, tuple(_cache_to_wire(snapshot) for snapshot in snapshots))
                    for ranks, snapshots in decision["caches"].items()
                ],
            )
            self._control.set(f"{prefix}/decision", msgspec.msgpack.encode(wire_decision))
        decision = msgspec.msgpack.decode(self._control.get(f"{prefix}/decision"))
        decision["caches"] = {
            tuple(ranks): tuple(_cache_from_wire(snapshot) for snapshot in snapshots)
            for ranks, snapshots in decision["caches"]
        }
        return decision

    def _decision(self, gathered):
        self._validate_domain(gathered)
        errors = [entry["error"] for entry in gathered if entry["error"]]
        error = min(errors, key=lambda item: int(item["rank"])) if errors else None
        stop = error is not None or any(entry["stop"] for entry in gathered)
        if self._control is not None:
            stop |= self._control.check(["stop"])
        tuning = () if stop else _authorize_tuning(gathered, self.world_ranks)
        caches = {} if stop else _authorize_caches(gathered, self.world_ranks)
        if not stop:
            authorized = {item[0] for item in tuning}
            pending = {item[0] for entry in gathered for item in entry["tuning"]}
            if authorized != pending:
                raise RuntimeError("preparation ranks reached incompatible tuning boundaries")
        return dict(
            stop=stop,
            error=error,
            collective=None if error else _authorize_ready(gathered, self.world_ranks),
            tuning=tuning,
            caches=caches,
            done=all(entry["local_done"] for entry in gathered)
            and (error is None or all(entry["cleanup_complete"] for entry in gathered)),
        )

    def abort(self) -> dict[str, object]:
        """Stop optional work and close an active local preparation job."""
        self._stop = True
        self._authorized_key = None
        self._safe_close()
        self._local_done = True
        return self._outcome()

    def _advance_local(self) -> None:
        if self._local_done or self._error is not None:
            return
        job = self._job
        if job is None:
            raise RuntimeError("active preparation has no job")
        if self._stop:
            job.session.cancel_tuning()
        kwargs = dict(collective_key=self._authorized_key, tuning=self._authorized_tuning)
        if self._authorized_cache is not None:
            kwargs["cache"] = self._authorized_cache
        progress = job.advance(**kwargs)
        self._authorized_key = None
        self._authorized_tuning = None
        self._authorized_cache = None
        self._last_progress = progress
        if progress.pending_compilation:
            pool = job.session._pool
            if pool is not None:
                pool.wait_for_progress(timeout=0.05)
        if not progress.done:
            return

        job.result().close()
        self._reserve_workspace()
        self._active_requests = ()
        self._job = None
        if self._batches:
            self._begin_next_batch()
        else:
            self._local_done = True

    def _reserve_workspace(self) -> None:
        """Reserve winning plans' scratch on their serving lanes and streams."""
        if self._workspace is None:
            return
        from aphrodite.utils.torch_utils import aux_stream
        from aphrodite.v1.worker.workspace import use_workspace_lane

        for request in self._active_requests:
            specs = tuple(request.plan.scratch_specs())
            if not specs:
                continue
            shapes = tuple((spec.shape, spec.dtype) for spec in specs)
            for lane in self._request_workspace_lanes[request.name]:
                with use_workspace_lane(lane):
                    self._workspace.get_simultaneous(*shapes)
                    if request.name in self._auxiliary_stream_requests:
                        stream = aux_stream()
                        if stream is not None:
                            self._workspace.get_stream_simultaneous(stream.stream_id, *shapes)
        self._workspace.reserve_by_lane()

    def _ready(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        if self._last_progress is None:
            return ()
        return tuple((item.key, item.ranks) for item in self._last_progress.ready_collectives)

    def _ready_tuning(self) -> tuple[_TuningResult, ...]:
        if self._last_progress is None:
            return ()
        return tuple(
            (
                _scoped_key(item.key, item.ranks),
                item.ranks,
                None if item.assignment is None else item.assignment.to_dict(),
                item.latency_us,
                item.candidate_index,
                item.rejected_count,
                item.cute_programs,
            )
            for item in getattr(self._last_progress, "ready_tuning", ())
        )

    def _payload(self) -> dict[str, object]:
        tuning = self._ready_tuning()
        return {
            "round": self._round,
            "global_rank": self.global_rank,
            "world_ranks": self.world_ranks,
            "stop": self._stop,
            "ready": self._ready(),
            "tuning": tuning,
            "cache": _cache_to_wire(self._ready_cache()),
            "local_done": self._local_done,
            "error": self._error,
            "cleanup_complete": self._cleanup_complete,
        }

    def _ready_cache(self):
        return getattr(self._last_progress, "ready_cache", None)

    def _validate_domain(self, gathered: list[dict[str, object]]) -> None:
        if len(gathered) != len(self.world_ranks):
            raise RuntimeError("preparation control exchange did not include complete world domain")
        ranks = tuple(sorted(int(cast(int, entry["global_rank"])) for entry in gathered))
        if ranks != self.world_ranks or len(set(ranks)) != len(ranks):
            raise RuntimeError("preparation control exchange has inconsistent global ranks")
        for entry in gathered:
            if tuple(cast(tuple[int, ...], entry["world_ranks"])) != self.world_ranks:
                raise RuntimeError("preparation control exchange has inconsistent world domain")
            if entry["round"] != self._round:
                raise RuntimeError("preparation control exchange has inconsistent round")

    def _fail(self, error: BaseException) -> None:
        self._record_error(error)
        self._stop = True
        self._safe_close()
        if self._control is not None:
            self._control.set("failed", msgspec.msgpack.encode(self._error))
            self._control.set("stop", b"1")

    def _record_error(self, error: BaseException) -> None:
        self._error = {
            "rank": self.global_rank,
            "type": type(error).__name__,
            "message": str(error),
        }

    def _safe_close(self) -> None:
        if self._cleanup_complete:
            return
        primary = None
        if self._job is not None:
            try:
                self._job.close()
            except BaseException as error:
                primary = error
            self._job = None
        self._active_requests = ()
        self._batches.clear()
        self._cleanup_complete = True
        self._local_done = True
        if primary is not None and self._error is None:
            self._record_error(primary)

    def _outcome(self) -> dict[str, object]:
        return {
            "native": self._native,
            "round": self._round,
            "global_rank": self.global_rank,
            "done": self._global_done,
            "progress": self._last_progress,
            "error": self._error,
            "cleanup_complete": self._cleanup_complete,
        }


def _world_ranks(world_group) -> tuple[int, ...]:
    if world_group is None:
        return (0,)
    ranks = getattr(world_group, "ranks", None)
    if ranks is not None:
        ranks = tuple(ranks)
        if ranks and all(type(rank) is int and rank >= 0 for rank in ranks) and len(set(ranks)) == len(ranks):
            return tuple(sorted(ranks))
        raise ValueError("preparation control domain ranks must be unique nonnegative integers")
    size = getattr(world_group, "world_size", None)
    if type(size) is not int:
        cpu_group = getattr(world_group, "cpu_group", None)
        size = None if cpu_group is None else cpu_group.size()
    if type(size) is not int or size <= 0:
        raise ValueError("preparation control domain must expose global ranks")
    return tuple(range(size))


def _get_control_group(world_group) -> StatelessProcessGroup:
    """Return a store-backed channel isolated from model collectives."""
    tcp_group = getattr(world_group, "tcp_store_group", None)
    if tcp_group is not None:
        return tcp_group
    cpu_group = getattr(world_group, "cpu_group", None)
    get_store = getattr(cpu_group, "get_group_store", None)
    rank = getattr(world_group, "rank_in_group", None)
    world_size = getattr(world_group, "world_size", None)
    if get_store is None or type(rank) is not int or type(world_size) is not int or rank < 0 or rank >= world_size:
        raise RuntimeError("preparation control requires the world CPU group's metadata store")

    # The distributed world, not a process-global id cache, owns this channel.
    control_group = getattr(world_group, "_b12x_preparation_control", None)
    if control_group is None:
        import torch.distributed as dist

        from aphrodite.distributed.utils import StatelessProcessGroup

        control_group = StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=dist.PrefixStore("b12x_preparation_control_v1", get_store()),
        )
        world_group._b12x_preparation_control = control_group
    elif control_group.world_size != world_size:
        raise RuntimeError("preparation control world size changed for an active CPU group")
    return control_group


def _authorize_ready(
    gathered: list[dict[str, object]], world_ranks: tuple[int, ...]
) -> tuple[str, tuple[int, ...]] | None:
    ready_by_key: dict[str, set[int]] = {}
    participants_by_key: dict[str, tuple[int, ...]] = {}
    for entry in gathered:
        rank = int(cast(int, entry["global_rank"]))
        ready = cast(tuple[tuple[str, tuple[int, ...]], ...], entry["ready"])
        for key, ranks in ready:
            ranks = tuple(ranks)
            if ranks != tuple(sorted(set(ranks))) or not set(ranks) <= set(world_ranks):
                raise RuntimeError("preparation collective has an invalid participant set")
            previous = participants_by_key.setdefault(key, ranks)
            if previous != ranks:
                raise RuntimeError("preparation collective participants disagree for one key")
            ready_by_key.setdefault(key, set()).add(rank)
    choices = [key for key, ranks in participants_by_key.items() if ready_by_key.get(key) == set(ranks)]
    if not choices:
        return None
    key = min(choices)
    return key, participants_by_key[key]


def _authorize_caches(gathered, world_ranks):
    groups: dict[tuple[int, ...], dict[int, TuningCacheRequirement]] = {}
    for entry in gathered:
        cache = _cache_from_wire(entry.get("cache"))
        if cache is None:
            continue
        rank = int(entry["global_rank"])
        if rank not in cache.ranks or not set(cache.ranks) <= set(world_ranks):
            raise RuntimeError("preparation tuning cache has an invalid rank set")
        groups.setdefault(cache.ranks, {})[rank] = cache
    for ranks, snapshots in groups.items():
        if set(snapshots) != set(ranks):
            raise RuntimeError("preparation ranks reached incompatible tuning cache boundaries")
    return {ranks: tuple(snapshots[rank] for rank in ranks) for ranks, snapshots in groups.items()}


def _authorize_tuning(
    gathered: list[dict[str, object]], world_ranks: tuple[int, ...]
) -> tuple[tuple[object, ...], ...]:
    contributions: dict[str, list[tuple[float, int, int, dict[str, object], tuple[str, ...]]]] = {}
    ready_by_key: dict[str, set[int]] = {}
    participants_by_key: dict[str, tuple[int, ...]] = {}
    rejected_by_key: dict[str, int] = {}
    for entry in gathered:
        rank = int(cast(int, entry["global_rank"]))
        tuning = cast(tuple[_TuningResult, ...], entry.get("tuning", ()))
        for (
            key,
            ranks,
            assignment,
            latency_us,
            candidate_index,
            rejected,
            programs,
        ) in tuning:
            ranks = tuple(ranks)
            if ranks != tuple(sorted(set(ranks))) or not set(ranks) <= set(world_ranks) or rank not in ranks:
                raise RuntimeError("preparation tuning has an invalid rank set")
            previous = participants_by_key.setdefault(key, ranks)
            if previous != ranks:
                raise RuntimeError("preparation tuning participants disagree for one key")
            ready_by_key.setdefault(key, set()).add(rank)
            rejected_by_key[key] = rejected_by_key.get(key, 0) + rejected
            if assignment is not None:
                assert latency_us is not None and candidate_index is not None
                contributions.setdefault(key, []).append(
                    (
                        float(latency_us),
                        int(candidate_index),
                        rank,
                        assignment,
                        programs,
                    )
                )
    choices = [key for key, ranks in participants_by_key.items() if ready_by_key.get(key) == set(ranks)]
    winners = []
    for key in sorted(choices):
        if not contributions.get(key):
            raise RuntimeError(
                f"no launchable candidates for preparation tuning {key}: "
                f"{rejected_by_key[key]} candidates rejected across ranks "
                f"{participants_by_key[key]}"
            )
        candidates = contributions[key]
        indices = [candidate[1] for candidate in candidates]
        if len(indices) != len(set(indices)):
            raise RuntimeError("preparation tuning shards overlap")
        latency_us, candidate_index, source, assignment, programs = min(candidates)
        winners.append(
            (
                key,
                participants_by_key[key],
                assignment,
                latency_us,
                candidate_index,
                rejected_by_key[key],
                source,
                programs,
            )
        )
    return tuple(winners)
