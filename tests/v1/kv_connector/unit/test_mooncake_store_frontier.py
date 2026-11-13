# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import json
import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from sonar.distributed.kv_transfer.kv_connector.v1.mooncake_store.frontier import (
    FrontierStore,
    FrontierTransfer,
    locator_key,
)
from sonar.utils.hashing import sha256
from sonar.v1.core.hybrid_checkpoint.keys import frontier_key
from sonar.v1.core.kv_cache_utils import init_none_hash
from sonar.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test


class MemoryStore:
    """Execute scatter/gather copies against real CPU tensor addresses."""

    def __init__(self):
        self.values = {}
        self.failed_put = None
        self.exist_batches = []
        self.registrations = []

    def register_buffer(self, address, size):
        self.registrations.append((address, size))
        return 0

    def put(self, key, value):
        self.values.setdefault(key, bytes(value))
        return 0

    def get(self, key):
        return self.values.get(key, b"")

    def batch_is_exist(self, keys):
        self.exist_batches.append(list(keys))
        return [int(key in self.values) for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate_config):
        results = []
        for key, addrs, lengths in zip(keys, addresses, sizes, strict=True):
            if key == self.failed_put:
                results.append(-1)
            else:
                self.values.setdefault(
                    key, b"".join(ctypes.string_at(addr, size) for addr, size in zip(addrs, lengths, strict=True))
                )
                results.append(0)
        return results

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        results = []
        for key, addrs, lengths in zip(keys, addresses, sizes, strict=True):
            value = self.values.get(key)
            if value is None or len(value) != sum(lengths):
                results.append(-1)
                continue
            offset = 0
            for addr, size in zip(addrs, lengths, strict=True):
                ctypes.memmove(addr, value[offset : offset + size], size)
                offset += size
            results.append(offset)
        return results


def make_frontier(store, rank=0, world_size=1, *, prefix="test", dcp=1, num_blocks=8, dp_size=1):
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="test-model", dtype=torch.float32),
        cache_config=SimpleNamespace(cache_dtype="auto", mamba_cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            rank=rank,
            world_size=world_size * dp_size,
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=2,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["state"],
            MambaSpec(
                block_size=4,
                shapes=((4, 2),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
        KVCacheGroupSpec(
            ["ring"],
            CircularBufferSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=2,
                dtype=torch.float32,
            ),
        ),
    ]
    kv_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    frontier = FrontierStore(store, config, kv_config, prefix, 4)
    caches = {
        name: torch.arange(num_blocks * 8, dtype=torch.float32).reshape(num_blocks, 4, 2) + rank * 1000 + i * 100
        for i, name in enumerate(("attention", "state", "ring"))
    }
    frontier.register(caches, {})
    frontier.set_layout_namespace([frontier.schema] * world_size)
    return frontier, caches


def transfer():
    return FrontierTransfer("checkpoint", "locator", 7, [(0, 0, 1), (0, 1, 2), (1, 1, 3), (2, 0, 4)])


def lookup_saved(frontier, locator="locator", key="checkpoint"):
    candidates = [(length, key) for length in frontier.lookup_lengths([locator])]
    return frontier.lookup(candidates)


def assert_round_trip(store, replicate_config=None):
    prefix = uuid4().hex
    writer, source = make_frontier(store, prefix=prefix)
    saved = transfer()
    expected = {name: tensor.clone() for name, tensor in source.items()}
    writer.save(saved, replicate_config)
    assert lookup_saved(writer) == [(7, "checkpoint")]
    reader, destination = make_frontier(store, prefix=prefix, num_blocks=10)
    for tensor in destination.values():
        tensor.fill_(-1)
    restored = FrontierTransfer("checkpoint", "", 7, [(0, 0, 5), (0, 1, 6), (1, 1, 7), (2, 0, 8)])
    reader.load(restored)
    for name, src, dst in [("attention", 1, 5), ("attention", 2, 6), ("state", 3, 7), ("ring", 4, 8)]:
        torch.testing.assert_close(destination[name][dst], expected[name][src], rtol=0, atol=0)
    assert torch.all(destination["attention"][0] == -1)
    return writer, reader, source, destination


def test_frontier_scatter_gather_round_trip():
    assert_round_trip(MemoryStore())


def test_frontier_keeps_borrowed_registrations_out_of_store_ownership():
    store = MemoryStore()
    frontier, caches = make_frontier(store)
    store.registrations.clear()
    borrowed = {caches[name].untyped_storage().data_ptr() for name in ("attention", "state")}
    owned: dict[int, int] = {}
    frontier.register(caches, owned, borrowed)
    ring = caches["ring"].untyped_storage()
    assert store.registrations == [(ring.data_ptr(), ring.nbytes())]
    assert owned == {ring.data_ptr(): ring.nbytes()}
    assert not borrowed.intersection(owned)


def test_frontier_requires_all_rank_payloads():
    store = MemoryStore()
    rank0, rank0_caches = make_frontier(store, rank=0, world_size=2, dcp=2)
    rank1, rank1_caches = make_frontier(store, rank=1, world_size=2, dcp=2)
    rank0.save(transfer(), None)
    assert lookup_saved(rank0) == []
    rank1.save(transfer(), None)
    assert lookup_saved(rank0) == [(7, "checkpoint")]
    payload = rank1._payload_key("checkpoint", 1, 2, 0)
    del store.values[payload]
    assert lookup_saved(rank0) == []
    with pytest.raises(RuntimeError, match="payload load failed"):
        rank1.load(transfer())
    assert rank0_caches["state"].data_ptr() != rank1_caches["state"].data_ptr()


def test_global_rank_offset_uses_engine_local_manifests():
    store = MemoryStore()
    rank8, rank8_caches = make_frontier(store, rank=8, world_size=2, dp_size=5)
    rank9, rank9_caches = make_frontier(store, rank=9, world_size=2, dp_size=5)
    assert (rank8.rank, rank9.rank) == (0, 1)
    rank8.save(transfer(), None)
    assert lookup_saved(rank8) == []
    rank9.save(transfer(), None)
    assert lookup_saved(rank8) == [(7, "checkpoint")]
    assert rank8._manifest_key("checkpoint", 0) in store.values
    assert rank9._manifest_key("checkpoint", 1) in store.values
    assert rank8_caches["state"].data_ptr() != rank9_caches["state"].data_ptr()


def test_failed_put_does_not_publish_manifest():
    store = MemoryStore()
    frontier, _ = make_frontier(store)
    store.failed_put = frontier._payload_key("checkpoint", 0, 1, 1)
    with pytest.raises(RuntimeError, match="payload put failed"):
        frontier.save(transfer(), None)
    assert frontier._manifest_key("checkpoint", 0) not in store.values
    assert lookup_saved(frontier) == []


def test_frontier_namespace_and_layout_isolation():
    store = MemoryStore()
    writer, _ = make_frontier(store)
    writer.save(transfer(), None)
    other_deployment, _ = make_frontier(store, prefix="other-weights")
    other_dcp, _ = make_frontier(store, dcp=2)
    assert lookup_saved(other_deployment) == []
    assert lookup_saved(other_dcp) == []
    reader, caches = make_frontier(store)
    reader.register({name: tensor.transpose(1, 2) for name, tensor in caches.items()}, {})
    assert lookup_saved(reader) == []
    with pytest.raises(RuntimeError, match="layout mismatch"):
        reader.load(transfer())
    reader.set_layout_namespace([reader.schema])
    assert reader.prefix != writer.prefix
    assert lookup_saved(reader) == []


def test_immutable_locators_keep_distinct_lengths():
    store = MemoryStore()
    frontier, _ = make_frontier(store)
    frontier.save(transfer(), None)
    later = FrontierTransfer("later", "locator", 8, transfer().blocks)
    frontier.save(later, None)
    assert frontier.lookup_lengths(["locator"]) == [8, 7]
    assert frontier.lookup([(8, "later"), (7, "checkpoint")]) == [(8, "later"), (7, "checkpoint")]


def test_immutable_locator_restores_distinct_branches_at_same_length():
    store = MemoryStore()
    frontier, caches = make_frontier(store)
    frontier.save(transfer(), None)
    for tensor in caches.values():
        tensor.add_(1000)
    expected = caches["state"][3].clone()
    branch = FrontierTransfer("other-branch", "locator", 7, transfer().blocks)
    frontier.save(branch, None)
    # The old branch's immutable locator still discovers the new branch even
    # after the old manifest is evicted.
    del store.values[frontier._manifest_key("checkpoint", 0)]
    caches["state"].zero_()
    assert lookup_saved(frontier, key="other-branch") == [(7, "other-branch")]
    assert lookup_saved(frontier) == []
    frontier.load(branch)
    torch.testing.assert_close(caches["state"][3], expected, rtol=0, atol=0)


def test_presence_hint_limits_offset_probes_and_eviction_is_a_miss():
    store = MemoryStore()
    frontier, caches = make_frontier(store)
    locators = ["missing-a", "locator", "missing-b"]
    assert frontier.lookup_lengths(locators) == []
    assert [len(batch) for batch in store.exist_batches] == [3]
    frontier.save(transfer(), None)
    store.exist_batches.clear()
    assert frontier.lookup_lengths(locators) == [7]
    assert [len(batch) for batch in store.exist_batches[:2]] == [3, frontier.hash_block_size]
    del store.values[frontier._locator_hint_key("locator")]
    assert frontier._manifest_key("checkpoint", 0) in store.values
    assert frontier.lookup_lengths(locators) == []
    assert caches["state"].numel() > 0


def test_locator_never_hashes_uncomputed_sampled_token():
    init_none_hash(sha256)
    index = SimpleNamespace(hash_fn=sha256, hash_block_size=4)
    request = SimpleNamespace(
        block_hashes=[b"a" * 32, b"b" * 32],
        all_token_ids=list(range(9)),
        mm_features=[],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
    )
    before = locator_key(index, request, 8)
    request.all_token_ids[8] = 999
    assert locator_key(index, request, 8) == before
    assert before == frontier_key(sha256, request, 5, 4).hex()


def make_scheduler_frontier():
    from sonar.v1.request import RequestStatus

    from .test_mooncake_store_scheduler import _make_bare_scheduler

    init_none_hash(sha256)
    scheduler = _make_bare_scheduler(hash_block_size=4)
    scheduler.frontier_write_through = True
    scheduler._num_workers = 2
    blocks = scheduler._gpu_block_pool.get_new_blocks(4)
    request = SimpleNamespace(
        request_id="frontier",
        num_computed_tokens=11,
        num_tokens=12,
        resumable=False,
        status=RequestStatus.FINISHED_STOPPED,
        all_token_ids=list(range(12)),
        block_hashes=[b"a" * 32, b"b" * 32],
        mm_features=[],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
        get_max_cache_hit_length=lambda: 11,
    )
    managers = [
        SimpleNamespace(block_size=8, req_to_blocks={"frontier": blocks[:2]}),
        SimpleNamespace(block_size=4),
        SimpleNamespace(block_size=4),
    ]
    key = frontier_key(sha256, request, 11, 4)
    index = SimpleNamespace(
        pages=[0],
        states=[1],
        rings=[2],
        pool_size=4,
        hash_block_size=4,
        hash_fn=sha256,
        managers=managers,
        entries={key: SimpleNamespace(blocks={0: blocks[1], 1: blocks[2], 2: blocks[3]})},
    )
    scheduler.kv_cache_manager = SimpleNamespace(checkpoints=SimpleNamespace(frontiers=index, speculative=None))
    return scheduler, request, blocks, key


def test_frontier_save_pins_all_groups_until_every_rank_finishes():
    from sonar.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import MooncakeStoreWorkerMetadata

    scheduler, request, blocks, _ = make_scheduler_frontier()
    scheduler.register_finished_frontier(request)
    (meta,) = scheduler._finished_frontiers
    assert meta.frontier.blocks == [
        (0, 0, blocks[0].block_id),
        (0, 1, blocks[1].block_id),
        (1, 2, blocks[2].block_id),
        (2, 0, blocks[3].block_id),
    ]
    assert all(block.ref_cnt == 2 for block in blocks)
    scheduler._gpu_block_pool.free_blocks(blocks)
    assert all(block.ref_cnt == 1 for block in blocks)
    for remaining in (1, 0):
        scheduler.update_connector_output(
            SimpleNamespace(
                failed_recving=None,
                kv_connector_worker_meta=MooncakeStoreWorkerMetadata({meta.store_job_id: 1}),
            )
        )
        assert all(block.ref_cnt == remaining for block in blocks)
    assert not scheduler.has_pending_push_work()


@pytest.mark.parametrize("in_flight", [False, True])
def test_speculative_frontier_uses_published_stop_length(in_flight):
    scheduler, request, blocks, key = make_scheduler_frontier()
    scheduler.kv_cache_manager.checkpoints.speculative = object()
    request.num_in_flight_tokens = int(in_flight)
    if in_flight:
        # The mirrored stop has consumed every kept token, unlike an ordinary
        # sampled stop whose final token has not entered the model yet.
        request.num_tokens -= 1
        request.all_token_ids.pop()
    scheduler.register_finished_frontier(request)
    (meta,) = scheduler._finished_frontiers
    assert meta.frontier.num_tokens == 11
    assert meta.frontier.key == key.hex()
    assert all(block.ref_cnt == 2 for block in blocks)


@pytest.mark.parametrize("resumable", [False, True])
def test_abort_and_resumable_requests_do_not_queue_frontiers(resumable):
    from sonar.v1.request import RequestStatus

    scheduler, request, blocks, _ = make_scheduler_frontier()
    request.resumable = resumable
    if not resumable:
        request.status = RequestStatus.FINISHED_ABORTED
    scheduler.register_finished_frontier(request)
    assert not scheduler._finished_frontiers
    assert not scheduler.has_pending_push_work()
    assert all(block.ref_cnt == 1 for block in blocks)


def test_frontier_lookup_maps_sharded_pages_and_recurrent_destinations():
    from sonar.distributed.kv_transfer.kv_connector.v1.mooncake_store.connector import MooncakeStoreConnector
    from sonar.v1.core.kv_cache_manager import KVCacheBlocks

    scheduler, request, blocks, key = make_scheduler_frontier()
    scheduler.client.lookup_frontier_lengths = lambda req_id, locators: [11]
    scheduler.client.lookup_frontiers = lambda req_id, candidates: [
        (length, candidate) for length, candidate in candidates if candidate == key.hex()
    ]
    assert scheduler.get_num_new_matched_tokens(request, 8) == (3, True)
    connector = object.__new__(MooncakeStoreConnector)
    connector.connector_scheduler = scheduler
    assert connector.get_loaded_kv_cache_group_ids(request) == (0, 1, 2)
    null = scheduler._gpu_block_pool.null_block
    scheduler.update_state_after_alloc(
        request,
        KVCacheBlocks((blocks[:2], [null, null, blocks[2]], [blocks[3]])),
        3,
    )
    spec = scheduler.load_specs[request.request_id]
    assert spec.can_load
    assert spec.frontier.blocks == [
        (0, 1, blocks[1].block_id),
        (1, 2, blocks[2].block_id),
        (2, 0, blocks[3].block_id),
    ]
    # Length hints may come from another branch. Derive the identity from this
    # request before asking the store to validate the checkpoint.
    request.all_token_ids[10] = 999
    assert scheduler._lookup_frontier(request, 8) == 0


def make_controlled_frontier_client(monkeypatch, frontier):
    from .test_mooncake_store_connector import _make_lookup_client

    client, _ = _make_lookup_client()
    pending: deque[tuple[Future, Callable, tuple]] = deque()

    class NonblockingFuture(Future):
        def result(self, timeout=None):
            assert self.done() or timeout == 0, "scheduler waited for an unfinished lookup"
            return super().result(timeout)

    def submit(fn, *args):
        future = NonblockingFuture()
        pending.append((future, fn, args))
        return future

    def run_next():
        future, fn, args = pending.popleft()
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn(*args))
            except Exception as error:
                future.set_exception(error)

    def request(frames, deadline):
        query = json.loads(frames[1])
        result = (
            frontier.lookup_lengths(query["locators"]) if "locators" in query else frontier.lookup(query["candidates"])
        )
        return json.dumps(result).encode()

    monkeypatch.setattr(client.executor, "submit", submit)
    monkeypatch.setattr(client, "_request", request)
    return client, pending, run_next


@pytest.mark.parametrize("abort_phase", [None, "locators", "candidates"])
def test_frontier_lookup_pending_completion_and_abort(monkeypatch, abort_phase):
    from sonar.distributed.kv_transfer.kv_connector.v1.mooncake_store.connector import MooncakeStoreConnector
    from sonar.v1.request import RequestStatus

    scheduler, request, _, key = make_scheduler_frontier()
    index = scheduler.kv_cache_manager.checkpoints.frontiers
    frontier, _ = make_frontier(MemoryStore())
    frontier.save(
        FrontierTransfer(
            key.hex(),
            locator_key(index, request, 11),
            11,
            [(0, 0, 1), (0, 1, 2), (1, 2, 3), (2, 0, 4)],
        ),
        None,
    )
    client, pending, run_next = make_controlled_frontier_client(monkeypatch, frontier)
    scheduler.client = client
    connector = object.__new__(MooncakeStoreConnector)
    connector.connector_scheduler = scheduler
    try:
        assert scheduler.get_num_new_matched_tokens(request, 8) == (None, False)
        assert scheduler.get_num_new_matched_tokens(request, 8) == (None, False)
        assert len(pending) == 1
        assert request.request_id not in scheduler.load_specs
        if abort_phase != "locators":
            run_next()
            assert scheduler.get_num_new_matched_tokens(request, 8) == (None, False)
            assert len(pending) == 1
            assert request.request_id not in scheduler.load_specs
        if abort_phase is None:
            run_next()
            assert scheduler.get_num_new_matched_tokens(request, 8) == (3, True)
            assert scheduler.load_specs[request.request_id].frontier.key == key.hex()
            assert not pending
        else:
            phase_future = pending[0][0]
            # Admission can probe another request without waiting for this one.
            assert client.lookup_frontier_lengths("other", ["missing"]) is None
            other_future = pending[-1][0]
            # Abort also drops an ordinary prefetch belonging to this request.
            assert client.lookup(request.request_id, 12, request.block_hashes, non_block=True) is None
            ordinary_future = pending[-1][0]
            request.status = RequestStatus.FINISHED_ABORTED
            assert connector.request_finished_all_groups(request, ()) == (False, None)
            assert phase_future.cancelled() and ordinary_future.cancelled()
            assert not other_future.cancelled()
            assert set(client._lookups) == {("other", "locators")}
            run_next()  # Canceled frontier RPC is never sent.
            run_next()
            assert client.lookup_frontier_lengths("other", ["missing"]) == []
            assert request.request_id not in scheduler.load_specs
    finally:
        client.close()


def test_frontier_failed_receive_reports_request_and_recomputes():
    import queue

    from sonar.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import LoadSpec, ReqMeta

    from .test_mooncake_store_worker import _make_store_recving_thread

    store = MemoryStore()
    frontier, caches = make_frontier(store)
    frontier.save(transfer(), None)
    del store.values[frontier._payload_key("checkpoint", 0, 1, 1)]
    receiver = _make_store_recving_thread(store, is_hma_required=True)
    receiver.request_queue = queue.Queue()
    receiver.frontier_store = frontier
    meta = ReqMeta(
        req_id="frontier",
        token_len_chunk=0,
        block_ids=(),
        block_hashes=[],
        load_spec=LoadSpec(0, 7, True, frontier=transfer()),
    )
    receiver.add_request(meta)
    receiver._handle_request(receiver.request_queue.get())
    assert receiver.request_queue.unfinished_tasks == 0
    assert receiver.get_and_clear_finished_requests() == {"frontier"}
    failed = receiver.get_and_clear_failed_requests()
    assert failed == {"frontier"}
    scheduler, request, _, _ = make_scheduler_frontier()
    scheduler.update_connector_output(SimpleNamespace(failed_recving=failed, kv_connector_worker_meta=None))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert caches["state"].numel() > 0


def test_native_mooncake_frontier_round_trip():
    """Opt in against an existing native master, without clearing its store."""
    if not os.getenv("SONAR_TEST_MOONCAKE"):
        pytest.skip("Set SONAR_TEST_MOONCAKE=1 and MOONCAKE_CONFIG_PATH for native store I/O")
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    from sonar.distributed.mooncake_store import MooncakeStoreConfig, setup_mooncake_store
    from sonar.utils.network_utils import get_ip

    store = MooncakeDistributedStore()
    setup_mooncake_store(store, MooncakeStoreConfig.load_from_config(), get_ip())
    # Keep all registered tensors alive until the native store is closed.
    retained = []
    try:
        retained.append(assert_round_trip(store, ReplicateConfig()))
        writer, reader, source, destination = retained[0]
        for tensor in source.values():
            tensor.add_(1000)
        branch = FrontierTransfer("other-branch", "locator", 7, transfer().blocks)
        writer.save(branch, ReplicateConfig())
        assert lookup_saved(writer) == [(7, "checkpoint")]
        assert lookup_saved(writer, key="other-branch") == [(7, "other-branch")]
        reader.load(
            FrontierTransfer(
                "other-branch",
                "",
                7,
                [(0, 0, 5), (0, 1, 6), (1, 1, 7), (2, 0, 8)],
            )
        )
        for name, src, dst in [("attention", 1, 5), ("attention", 2, 6), ("state", 3, 7), ("ring", 4, 8)]:
            torch.testing.assert_close(destination[name][dst], source[name][src], rtol=0, atol=0)
        prefix = uuid4().hex
        rank0, caches0 = make_frontier(store, rank=0, world_size=2, dcp=2, prefix=prefix)
        rank1, caches1 = make_frontier(store, rank=1, world_size=2, dcp=2, prefix=prefix)
        retained.extend((rank0, rank1, caches0, caches1))
        rank0.save(transfer(), ReplicateConfig())
        assert lookup_saved(rank0) == []
        rank1.save(transfer(), ReplicateConfig())
        assert lookup_saved(rank0) == [(7, "checkpoint")]
        payload = rank1._payload_key("checkpoint", 1, 2, 0)
        assert store.remove(payload, force=True) == 0
        assert lookup_saved(rank0) == []
        with pytest.raises(RuntimeError, match="payload load failed"):
            rank1.load(transfer())
    finally:
        store.close()
    del retained
