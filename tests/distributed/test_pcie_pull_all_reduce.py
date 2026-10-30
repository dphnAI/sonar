# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import aphrodite.envs as envs
from aphrodite.distributed.device_communicators import cuda_communicator
from aphrodite.distributed.device_communicators import pcie_pull_all_reduce as pcie
from aphrodite.utils import import_utils


def _comm(monkeypatch, transport="host", rank=0):
    monkeypatch.setenv("APHRODITE_ALLREDUCE_PCIE_TRANSPORT", transport)
    monkeypatch.setattr(pcie.dist, "get_rank", lambda group: rank)
    monkeypatch.setattr(pcie.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(pcie, "_UNRETIRED_WORKSPACES", [])
    return pcie.PciePullAllReduce(object(), torch.device(f"cuda:{rank}"))


@pytest.mark.parametrize(
    ("transports", "selectors", "message"),
    [
        (["paired", "host", "paired", "host"], [True] * 4, "must agree on APHRODITE_ALLREDUCE_PCIE_TRANSPORT"),
        (["host", "automatic", "host", "host"], [True] * 4, "transport configuration failed: rank 1:"),
        (["host"] * 4, [True, False, True, True], "must agree on PCIe all-reduce backend selectors"),
    ],
)
def test_transport_disagreement_fails_every_rank_before_setup(monkeypatch, transports, selectors, message):
    selections = []
    for transport, enabled in zip(transports, selectors):
        monkeypatch.setenv("APHRODITE_ALLREDUCE_PCIE_TRANSPORT", transport)
        value, error = None, None
        if enabled:
            try:
                value = envs.APHRODITE_ALLREDUCE_PCIE_TRANSPORT
            except ValueError as exc:
                error = str(exc)
        selections.append((enabled, False, value, error))

    def base_init(self, cpu_group, *args, **kwargs):
        self.cpu_group = cpu_group
        self.world_size = 4

    def gather(entries, value, group):
        entries[:] = selections

    monkeypatch.setattr(cuda_communicator.DeviceCommunicatorBase, "__init__", base_init)
    monkeypatch.setattr(cuda_communicator.dist, "all_gather_object", gather)
    # Any successful departure from consensus would attempt backend setup.
    monkeypatch.setattr(pcie.PciePullAllReduce, "__init__", Mock(side_effect=AssertionError("setup began")))
    monkeypatch.setattr(
        "aphrodite.distributed.device_communicators.pynccl.PyNcclCommunicator",
        Mock(side_effect=AssertionError("setup began")),
    )
    for transport, enabled in zip(transports, selectors):
        monkeypatch.setenv("APHRODITE_ALLREDUCE_PCIE_TRANSPORT", transport)
        monkeypatch.setattr(envs, "APHRODITE_ALLREDUCE_USE_PCIE_PULL", enabled)
        monkeypatch.setattr(envs, "APHRODITE_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", False)
        with pytest.raises(ValueError, match=message):
            cuda_communicator.CudaCommunicator(object(), unique_name="tp:0")


def _dispatcher(monkeypatch, comm):
    dispatcher = cuda_communicator.CudaCommunicator.__new__(cuda_communicator.CudaCommunicator)
    dispatcher.pcie_pull_ar_comm = comm
    dispatcher.world_size = 4
    dispatcher.fi_ar_comm = dispatcher.fi_pcie_ipc_ar_comm = None
    dispatcher.qr_comm = dispatcher.ca_comm = dispatcher.symm_mem_comm = dispatcher.aiter_ar_comm = None
    dispatcher.use_aiter_allreduce = False
    dispatcher.pynccl_comm = Mock(disabled=False, world_size=4)
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_allreduce", lambda *args: False)
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)
    monkeypatch.setattr(envs, "APHRODITE_BATCH_INVARIANT", False)
    return dispatcher


@pytest.mark.parametrize(
    ("transport", "rows", "width", "dtype", "selected"),
    [
        ("paired", 8, 5120, torch.bfloat16, True),
        ("paired", 8, 6144, torch.bfloat16, True),
        ("paired", 9, 5120, torch.bfloat16, False),
        ("host", 1, 5120, torch.bfloat16, True),
        ("host", 32, 5120, torch.bfloat16, True),
        ("host", 33, 5120, torch.bfloat16, True),
        ("host", 64, 5120, torch.bfloat16, True),
        ("host", 65, 5120, torch.bfloat16, False),
        ("host", 1, 6144, torch.bfloat16, False),
        ("host", 1, 5120, torch.float16, False),
    ],
)
def test_reduce_dispatch_limits_and_faults_never_fallback(monkeypatch, transport, rows, width, dtype, selected):
    comm = _comm(monkeypatch, transport)
    comm.device = torch.device("cpu")
    comm._ptr, comm.disabled = 1, False
    comm._native = SimpleNamespace(all_reduce=Mock(side_effect=RuntimeError("terminal native fault")))
    monkeypatch.setattr(pcie.current_platform, "is_current_stream_capturing", lambda: False)
    dispatcher = _dispatcher(monkeypatch, comm)
    inp = torch.zeros((rows, width), dtype=dtype)
    fallback = torch.full_like(inp, 7)
    dispatcher.pynccl_comm.all_reduce.return_value = fallback

    if selected:
        with pytest.raises(RuntimeError, match="terminal native fault"):
            dispatcher.all_reduce(inp)
        # A sticky failure cannot change backend selection on the next call.
        with pytest.raises(RuntimeError, match="terminally failed"):
            dispatcher.all_reduce(inp)
        dispatcher.pynccl_comm.all_reduce.assert_not_called()
    else:
        assert dispatcher.all_reduce(inp) is fallback


@pytest.mark.parametrize(
    ("transport", "size", "selected"),
    [("paired", 16384, True), ("host", 1024, True), ("host", 1025, False)],
)
def test_gather_dispatch_limits_and_rank_order(monkeypatch, transport, size, selected):
    comm = _comm(monkeypatch, transport)
    comm.device = torch.device("cpu")
    comm._ptr, comm.disabled = 1, False

    def gather(pointer, inp, output):
        output.copy_(torch.cat([inp + rank for rank in range(4)]))

    comm._native = SimpleNamespace(all_gather=gather)
    monkeypatch.setattr(pcie.current_platform, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cuda_communicator.current_platform, "is_rocm", lambda: False)
    dispatcher = _dispatcher(monkeypatch, comm)
    inp = torch.zeros((1, size), dtype=torch.uint8)
    if selected:
        result = dispatcher.all_gather(inp, dim=1)
        assert torch.equal(result, torch.cat([inp + rank for rank in range(4)], dim=1))
    else:
        monkeypatch.setattr(
            cuda_communicator.DeviceCommunicatorBase,
            "all_gather",
            lambda self, inp, dim: torch.full((1, 4 * size), 9, dtype=inp.dtype),
        )
        assert torch.all(dispatcher.all_gather(inp, dim=1) == 9)


def _nvml(*, duplicate=False):
    return SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: f"GPU-{0 if duplicate else handle}",
        nvmlDeviceGetTopologyCommonAncestor=Mock(side_effect=AssertionError("NUMA eligibility was consulted")),
        nvmlDeviceGetP2PStatus=Mock(side_effect=AssertionError("P2P eligibility was consulted")),
    )


def test_host_topology_allows_no_p2p_and_cross_numa_but_requires_distinct_gpus(monkeypatch):
    comm = _comm(monkeypatch)
    nvml = _nvml()
    monkeypatch.setattr(import_utils, "import_pynvml", lambda: nvml)
    assert comm._topology([0, 1, 2, 3])[0] == tuple(f"GPU-{rank}" for rank in range(4))
    nvml = _nvml(duplicate=True)
    with pytest.raises(ValueError, match="different physical GPU"):
        comm._topology([0, 0, 2, 3])


def _config():
    parallel = SimpleNamespace(tensor_parallel_size=4, disable_custom_all_reduce=False)
    for field in (
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "data_parallel_size",
        "data_parallel_size_local",
        "nnodes",
    ):
        setattr(parallel, field, 1)
    for field in (
        "enable_dbo",
        "use_ubatching",
        "enable_expert_parallel",
        "enable_eplb",
        "enable_elastic_ep",
        "enable_fault_tolerance",
        "enable_batch_sharded_sampling",
    ):
        setattr(parallel, field, False)
    config = SimpleNamespace(
        parallel_config=parallel,
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            runner_type="generate",
            dtype=torch.bfloat16,
            get_hidden_size=lambda: 5120,
            architectures=["DeepseekV4ForCausalLM"],
            model="deepseek",
            revision=None,
            quantization=None,
        ),
        aux_output_config=SimpleNamespace(enabled=False),
        cache_config=SimpleNamespace(kv_offloading_size=None),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 8, 16, 32, 64], cudagraph_mode="FULL"),
    )
    for field in (
        "lora_config",
        "kv_transfer_config",
        "ec_transfer_config",
        "weight_transfer_config",
        "kv_events_config",
        "diffusion_config",
    ):
        setattr(config, field, None)
    return config


class _CpuCollective:
    """Threaded Gloo stand-in; mappings and file lifetime remain real OS resources."""

    def __init__(self):
        self.barrier = threading.Barrier(4, timeout=10)
        self.values = [None] * 4

    def gather(self, entries, value, group):
        self.values[group.rank] = value
        self.barrier.wait()
        entries[:] = self.values
        self.barrier.wait()


class _ColdNative:
    """No GPU emulation: only the cold registration/context ownership contract."""

    def __init__(self):
        self.contexts = {}
        self.registrations = set()
        self.fail_init_rank = None
        self.fail_dispose_rank = None
        self.fail_registration_retained = False
        self.fail_unregister = False

    def buffer_size(self, capacity, blocks, transport):
        # Size-dependent stand-in for controls and two BF16 payload parities.
        return 65536 + 2 * capacity * 2

    def register_host_buffer(self, address, length):
        assert ctypes.string_at(address, length) == bytes(length)
        self.registrations.add(address)
        if self.fail_registration_retained:
            raise RuntimeError("PCIE_HOST_REGISTRATION_RETAINED: CUDA registration rollback failed")
        return address

    def unregister_host_buffer(self, address):
        assert address not in self.contexts.values()
        if self.fail_unregister:
            raise RuntimeError("CUDA host unregister failed")
        self.registrations.remove(address)

    def init(self, peers, state, rank, capacity, blocks, threads, timeout_ns, transport):
        if rank == self.fail_init_rank:
            raise RuntimeError("native initialization rejected")
        self.contexts[rank + 1] = peers[0]
        return rank + 1

    def bind_stream(self, pointer):
        pass

    def local_status(self, pointer):
        return 0

    def dispose(self, pointer):
        if pointer - 1 == self.fail_dispose_rank:
            raise RuntimeError("CUDA stream drain failed")
        del self.contexts[pointer]

    all_reduce = all_gather = close_step = None


def _run(comms, operation):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(operation, comm) for comm in comms]
        return [future.exception(timeout=20) for future in futures]


@pytest.fixture
def host_cluster(monkeypatch, tmp_path):
    collective = _CpuCollective()
    native = _ColdNative()
    monkeypatch.setenv("APHRODITE_ALLREDUCE_PCIE_TRANSPORT", "host")
    monkeypatch.setattr(envs, "APHRODITE_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", False)
    monkeypatch.setattr(pcie, "_UNRETIRED_WORKSPACES", [])
    monkeypatch.setattr(pcie.dist, "get_rank", lambda group: group.rank)
    monkeypatch.setattr(pcie.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(pcie.dist, "get_backend", lambda group: pcie.dist.Backend.GLOO)
    monkeypatch.setattr(pcie.dist, "monitored_barrier", lambda group, **kwargs: collective.barrier.wait())
    monkeypatch.setattr(pcie.dist, "all_gather_object", collective.gather)
    monkeypatch.setattr(pcie, "in_the_same_node_as", lambda group, source_rank: [True] * 4)
    monkeypatch.setattr(import_utils, "import_pynvml", _nvml)
    monkeypatch.setattr(pcie, "gpu_sync_allowed", nullcontext)
    monkeypatch.setattr(pcie, "current_stream", lambda: None)
    monkeypatch.setattr(
        pcie,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda index: SimpleNamespace(to_int=lambda: 120),
            visible_device_id_to_physical_device_id=lambda index: index,
            stream=lambda stream: nullcontext(),
        ),
    )
    monkeypatch.setattr(torch.ops, "_C_pcie_pull", native)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda index: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: None)
    monkeypatch.setattr(pcie.PciePullAllReduce, "_drain_bound_stream", lambda self: None)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: empty(*args, **(kwargs | {"device": "cpu"})))
    mkstemp = tempfile.mkstemp
    monkeypatch.setattr(pcie.tempfile, "mkstemp", lambda **kwargs: mkstemp(**(kwargs | {"dir": tmp_path})))
    comms = [pcie.PciePullAllReduce(SimpleNamespace(rank=rank), torch.device(f"cuda:{rank}")) for rank in range(4)]
    try:
        yield comms, native, tmp_path
    finally:
        native.fail_dispose_rank = None
        native.fail_unregister = False
        if any(comm._host_fd is not None for comm in comms):
            assert _run(comms, lambda comm: comm._release_resources()) == [None] * 4


def test_host_cold_setup_shares_payload_and_retires_without_ipc(host_cluster):
    comms, native, directory = host_cluster
    assert _run(comms, lambda comm: comm.initialize(_config(), use_v2_model_runner=True)) == [None] * 4
    mappings = [comm._host_mapping for comm in comms]
    descriptors = [comm._host_fd for comm in comms]
    slab_bytes = native.buffer_size(64 * 5120, 1, 3)
    assert all(len(mapping) == 4 * slab_bytes for mapping in mappings)
    assert all(os.fstat(descriptor).st_size == 4 * slab_bytes for descriptor in descriptors)
    for rank, comm in enumerate(comms):
        ctypes.c_uint64.from_address(comm._local_ptr).value = rank + 11
    for comm in comms:
        assert comm.initialized
        assert [ctypes.c_uint64.from_address(pointer).value for pointer in comm._peer_ptrs] == [11, 12, 13, 14]
    assert not list(directory.iterdir())  # The live inode is owned by mappings/descriptors.
    assert _run(comms, lambda comm: comm.destroy()) == [None] * 4
    assert all(mapping.closed for mapping in mappings)
    assert not native.registrations and not native.contexts
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_host_native_initialization_failure_rolls_back_collectively(host_cluster):
    comms, native, directory = host_cluster
    native.fail_init_rank = 1
    errors = _run(comms, lambda comm: comm.initialize(_config(), use_v2_model_runner=True))
    assert all(isinstance(error, RuntimeError) and "native initialization rejected" in str(error) for error in errors)
    assert all(not comm.initialized and comm._host_fd is None for comm in comms)
    assert not native.registrations and not native.contexts
    assert not list(directory.iterdir())


@pytest.mark.parametrize("unsafe", ["graph", "drain"])
def test_host_unsafe_retirement_keeps_registered_mappings(host_cluster, unsafe):
    comms, native, _ = host_cluster
    assert _run(comms, lambda comm: comm.initialize(_config(), use_v2_model_runner=True)) == [None] * 4
    if unsafe == "graph":
        lease = pcie._GraphLease(comms[1])
        comms[1]._graph_leases.add(lease)
        message = "release every target/draft CUDA graph"
    else:
        native.fail_dispose_rank = 1
        message = "CUDA stream drain failed"
    errors = _run(comms, lambda comm: comm.destroy())
    assert all(isinstance(error, RuntimeError) and message in str(error) for error in errors)
    assert all(comm._pinned and comm._host_registered and not comm._host_mapping.closed for comm in comms)
    for comm in comms:
        ctypes.c_uint64.from_address(comm._host_address).value = 123
        assert os.fstat(comm._host_fd).st_size > 0


@pytest.mark.parametrize("ineligible", ["different_host", "tp2", "sm90", "missing_registration"])
def test_host_preserves_non_transport_eligibility_requirements(host_cluster, monkeypatch, ineligible):
    comms, native, directory = host_cluster
    config = _config()
    if ineligible == "different_host":
        monkeypatch.setattr(pcie, "in_the_same_node_as", lambda group, source_rank: [True, False, True, True])
        message = "same node"
    elif ineligible == "tp2":
        config.parallel_config.tensor_parallel_size = 2
        message = "must both be four"
    elif ineligible == "sm90":
        monkeypatch.setattr(
            pcie.current_platform, "get_device_capability", lambda index: SimpleNamespace(to_int=lambda: 90)
        )
        message = "only on SM120"
    else:
        monkeypatch.delattr(_ColdNative, "register_host_buffer")
        message = "register_host_buffer is unavailable"
    errors = _run(comms, lambda comm: comm.initialize(config, use_v2_model_runner=True))
    assert all(isinstance(error, (ValueError, RuntimeError)) and message in str(error) for error in errors)
    assert all(not comm.initialized and comm._host_fd is None for comm in comms)
    assert not native.registrations and not native.contexts
    assert not list(directory.iterdir())


def test_host_rejects_a_different_shared_file_namespace_collectively(host_cluster, monkeypatch):
    comms, native, directory = host_cluster
    fstat = os.fstat

    def different_inode(descriptor):
        stat = fstat(descriptor)
        if descriptor == comms[1]._host_fd:
            return SimpleNamespace(st_dev=stat.st_dev, st_ino=stat.st_ino + 1, st_size=stat.st_size)
        return stat

    monkeypatch.setattr(pcie.os, "fstat", different_inode)
    errors = _run(comms, lambda comm: comm.initialize(_config(), use_v2_model_runner=True))
    assert all(isinstance(error, RuntimeError) and "shared file namespace" in str(error) for error in errors)
    assert all(not comm.initialized and comm._host_fd is None for comm in comms)
    assert not native.registrations and not native.contexts
    assert not list(directory.iterdir())


@pytest.mark.parametrize("retry_fails", [False, True])
def test_host_retained_registration_error_retries_or_keeps_mappings(host_cluster, retry_fails):
    comms, native, directory = host_cluster
    native.fail_registration_retained = True
    native.fail_unregister = retry_fails
    errors = _run(comms, lambda comm: comm.initialize(_config(), use_v2_model_runner=True))
    assert all(isinstance(error, RuntimeError) and "PCIE_HOST_REGISTRATION_RETAINED" in str(error) for error in errors)
    if retry_fails:
        assert all(comm._pinned and comm._host_registered and not comm._host_mapping.closed for comm in comms)
        assert native.registrations
    else:
        assert all(not comm.initialized and comm._host_fd is None for comm in comms)
        assert not native.registrations
    assert not native.contexts
    assert not list(directory.iterdir())
