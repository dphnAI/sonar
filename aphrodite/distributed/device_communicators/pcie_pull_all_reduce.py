# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-lane, fixed-order TP4 paired all-reduce using CUDA IPC slabs."""

import gc
import threading
import time
import weakref
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import aphrodite.envs as envs
from aphrodite import _custom_ops as ops
from aphrodite.distributed.parallel_state import in_the_same_node_as
from aphrodite.logger import init_logger
from aphrodite.platforms import current_platform
from aphrodite.utils.gpu_sync_debug import gpu_sync_allowed
from aphrodite.utils.torch_utils import current_stream
from aphrodite.v1.worker.workspace import retain_cuda_graph_capture_resource

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig

logger = init_logger(__name__)

# A failed or uncoordinated shutdown must not return live IPC/state storage to
# an allocator. Only process exit may reclaim these sessions.
_UNRETIRED_WORKSPACES: list["PciePullAllReduce"] = []


class _GraphLease:
    def __init__(self, owner: "PciePullAllReduce") -> None:
        self.owner = owner


class PciePullAllReduce:
    """Explicitly initialized communicator; runtime failures never select NCCL.

    Initialization, readiness checks and healthy retirement use the Gloo TP
    group. Captured graphs retain a lease until they are released. Faulted
    retirement is local and pins the workspace until process exit; neither
    the destructor nor an unsuccessful teardown frees peer mappings.
    """

    MAX_ROWS = 8
    WIDTHS = (5120, 6144)
    CAPACITY = MAX_ROWS * 6144
    BLOCKS = 1
    THREADS = 512
    TRANSPORT = 2
    TIMEOUT_NS = 2_000_000_000
    HIDDEN_SIZE = 5120
    # Per-rank payload limit of the one-shot gather (native kGatherBytes).
    GATHER_BYTES = 16384
    CONTROL_TIMEOUT = timedelta(seconds=30)

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.disabled = True
        self._ptr = 0
        self._local_ptr = 0
        self._peer_ptrs: list[int] = [0] * self.world_size
        self._ipc_handles: list[Any] = []
        self._state: torch.Tensor | None = None
        self._native: Any = None
        self._failed = False
        self._destroyed = False
        self._pinned = False
        self._initialization_attempted = False
        self._capturing = False
        self._capture_lease: _GraphLease | None = None
        self._graph_leases: weakref.WeakSet[_GraphLease] = weakref.WeakSet()
        self._lock = threading.RLock()
        self._bound_stream: Any = None

    @property
    def initialized(self) -> bool:
        return bool(self._ptr and not self.disabled)

    def _barrier(self) -> None:
        dist.monitored_barrier(
            group=self.group,
            timeout=self.CONTROL_TIMEOUT,
            wait_all_ranks=True,
        )

    def _exchange(self, phase: str, value: Any = None, error: str | None = None) -> list[Any]:
        self._barrier()
        entries: list[Any] = [None] * self.world_size
        dist.all_gather_object(entries, (phase, error, value), group=self.group)
        errors = [f"rank {rank}: {entry[1]}" for rank, entry in enumerate(entries) if entry[1] is not None]
        if any(entry[0] != phase for entry in entries):
            errors.append("ranks entered different lifecycle phases")
        if errors:
            raise RuntimeError(f"PCIe paired all-reduce {phase} failed: " + "; ".join(errors))
        return [entry[2] for entry in entries]

    def _stage(self, phase: str, operation):
        value = None
        error = None
        try:
            value = operation()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return self._exchange(phase, value, error)

    def _manifest(self, config: "AphroditeConfig", use_v2_model_runner: bool) -> dict[str, Any]:
        parallel = config.parallel_config
        model = config.model_config
        errors: list[str] = []
        if not use_v2_model_runner:
            errors.append("the V2 model runner is required")
        if not current_platform.is_cuda() or self.device.type != "cuda" or self.device.index is None:
            errors.append("an indexed NVIDIA CUDA device is required")
        if self.world_size != 4 or parallel.tensor_parallel_size != 4:
            errors.append("tensor_parallel_size and TP group size must both be four")
        for field in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
            "data_parallel_size",
            "data_parallel_size_local",
            "nnodes",
        ):
            if getattr(parallel, field) != 1:
                errors.append(f"{field}=1 is required")
        for field in (
            "enable_dbo",
            "use_ubatching",
            "enable_expert_parallel",
            "enable_eplb",
            "enable_elastic_ep",
            "enable_fault_tolerance",
            "enable_batch_sharded_sampling",
        ):
            if getattr(parallel, field):
                errors.append(f"{field} is unsupported")
        if model.enable_sleep_mode:
            errors.append("sleep/migration is unsupported")
        if model.runner_type == "pooling":
            errors.append("pooling runners do not provide the collective output-close gate")
        for field in (
            "lora_config",
            "kv_transfer_config",
            "ec_transfer_config",
            "weight_transfer_config",
            "kv_events_config",
            "diffusion_config",
        ):
            if getattr(config, field) is not None:
                errors.append(f"{field} is unsupported")
        if config.aux_output_config.enabled:
            errors.append("external auxiliary outputs are unsupported")
        if config.cache_config.kv_offloading_size is not None:
            errors.append("KV offloading is unsupported")
        if config.offload_config.uva.cpu_offload_gb or config.offload_config.prefetch.offload_group_size:
            errors.append("weight offloading is unsupported")
        if model.dtype != torch.bfloat16 or model.get_hidden_size() != self.HIDDEN_SIZE:
            errors.append("the initial model invariant is BF16 with hidden_size=5120")
        if envs.APHRODITE_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:
            errors.append("FlashInfer PCIe IPC and PCIe paired all-reduce cannot both be selected")

        speculative = config.speculative_config
        draft_manifest: tuple[Any, ...] | None = None
        if speculative is not None:
            if speculative.enable_adaptive_verification:
                errors.append("adaptive verification can diverge rank-local collective sizes after quarantine")
            draft = speculative.draft_model_config
            draft_tp = speculative.draft_tensor_parallel_size
            if draft is not None:
                if draft.dtype != torch.bfloat16 or draft.get_hidden_size() != self.HIDDEN_SIZE:
                    errors.append("draft model must also use BF16 and hidden_size=5120")
                if draft_tp != 4:
                    errors.append("draft model must share the TP4 group")
                draft_manifest = (
                    draft_tp,
                    str(draft.dtype),
                    draft.get_hidden_size(),
                    draft.model,
                    draft.revision,
                    draft.quantization,
                )
            draft_manifest = (speculative.method, speculative.num_speculative_tokens, draft_manifest)

        if errors:
            raise ValueError("; ".join(errors))
        capability = current_platform.get_device_capability(self.device.index)
        if capability is None or capability.to_int() != 120:
            raise ValueError("PCIe paired all-reduce is supported only on SM120")
        self._native = torch.ops._C_pcie_pull
        for name in (
            "buffer_size",
            "init",
            "all_reduce",
            "all_gather",
            "close_step",
            "bind_stream",
            "local_status",
            "dispose",
            "close_ipc_handle",
        ):
            if not hasattr(self._native, name):
                raise RuntimeError(f"native operator _C_pcie_pull.{name} is unavailable")
        physical_id = current_platform.visible_device_id_to_physical_device_id(self.device.index)
        return {
            "policy": (
                self.CAPACITY,
                self.BLOCKS,
                self.THREADS,
                self.TRANSPORT,
                self.MAX_ROWS,
                self.WIDTHS,
                self.TIMEOUT_NS,
                self.GATHER_BYTES,
                self.HIDDEN_SIZE,
                str(model.dtype),
                tuple(model.architectures),
                model.model,
                model.revision,
                model.quantization,
                draft_manifest,
                tuple(config.compilation_config.cudagraph_capture_sizes or ()),
                str(config.compilation_config.cudagraph_mode),
                bool(envs.APHRODITE_BATCH_INVARIANT),
                bool(envs.APHRODITE_ALLREDUCE_USE_SYMM_MEM),
                bool(envs.APHRODITE_ALLREDUCE_USE_FLASHINFER),
                parallel.disable_custom_all_reduce,
                self._native.buffer_size(self.CAPACITY, self.BLOCKS, self.TRANSPORT),
            ),
            "physical_id": physical_id,
            "capability": capability.to_int(),
        }

    def _topology(self, physical_ids: list[int]) -> tuple[Any, ...]:
        # NVML READ/WRITE capabilities and successful IPC imports are necessary
        # setup gates, not a substitute for qualifying SM-issued transport.
        from aphrodite.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            handles = [pynvml.nvmlDeviceGetHandleByIndex(device) for device in physical_ids]
            uuids = tuple(str(pynvml.nvmlDeviceGetUUID(handle)) for handle in handles)
            if len(set(uuids)) != 4:
                raise ValueError("each TP rank must own a different physical GPU")
            topology = []
            for source, handle in enumerate(handles):
                for target, peer_handle in enumerate(handles):
                    if source == target:
                        continue
                    ancestor = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle, peer_handle)
                    if ancestor > pynvml.NVML_TOPOLOGY_NODE:
                        raise ValueError(f"GPU route {source}->{target} crosses a CPU/NUMA system link")
                    for capability in (pynvml.NVML_P2P_CAPS_INDEX_READ, pynvml.NVML_P2P_CAPS_INDEX_WRITE):
                        status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, capability)
                        if status != pynvml.NVML_P2P_STATUS_OK:
                            raise ValueError(f"GPU route {source}->{target} lacks CUDA peer read/write capability")
                    topology.append((source, target, ancestor))
            return (uuids, tuple(topology))
        finally:
            pynvml.nvmlShutdown()

    def initialize(self, config: "AphroditeConfig", *, use_v2_model_runner: bool) -> None:
        """Collectively allocate the fixed workspace before GPU memory profiling."""
        if dist.get_backend(self.group) != dist.Backend.GLOO:
            raise ValueError("PCIe paired all-reduce requires a CPU Gloo process group")
        with self._lock, gpu_sync_allowed():

            def manifest():
                if self._initialization_attempted or self._destroyed:
                    raise RuntimeError("a PCIe paired workspace cannot be initialized twice or revived")
                return self._manifest(config, use_v2_model_runner)

            manifests = self._stage("configuration", manifest)
            self._initialization_attempted = True
            if any(entry["policy"] != manifests[0]["policy"] for entry in manifests):
                raise ValueError("PCIe paired all-reduce setup/admission policy differs between TP ranks")
            same_node = self._stage("node locality", lambda: all(in_the_same_node_as(self.group, source_rank=0)))
            if not all(same_node):
                raise ValueError("PCIe paired all-reduce requires every TP rank on the same node")
            physical_ids = [entry["physical_id"] for entry in manifests]
            topologies = self._stage("topology", lambda: self._topology(physical_ids))
            if any(topology != topologies[0] for topology in topologies):
                raise ValueError("PCIe paired all-reduce topology differs between TP ranks")
            with torch.accelerator.device_index(self.device.index):
                try:

                    def allocate():
                        size = self._native.buffer_size(self.CAPACITY, self.BLOCKS, self.TRANSPORT)
                        self._state = torch.empty(8, dtype=torch.int64, device=self.device)
                        self._local_ptr, handle = ops.allocate_shared_buffer_and_handle(size)
                        self._peer_ptrs[self.rank] = self._local_ptr
                        return handle

                    self._ipc_handles = self._stage("allocation", allocate)

                    def open_peers():
                        for peer, handle in enumerate(self._ipc_handles):
                            if peer != self.rank:
                                self._peer_ptrs[peer] = ops.open_mem_handle(handle)

                    self._stage("IPC import", open_peers)

                    def initialize_native():
                        self._bound_stream = current_stream()
                        with current_platform.stream(self._bound_stream):
                            self._ptr = self._native.init(
                                self._peer_ptrs,
                                self._state,
                                self.rank,
                                self.CAPACITY,
                                self.BLOCKS,
                                self.THREADS,
                                self.TIMEOUT_NS,
                                self.TRANSPORT,
                            )
                            self._native.bind_stream(self._ptr)
                        torch.accelerator.synchronize(self.device)

                    self._stage("native initialization", initialize_native)
                except BaseException:
                    self._failed = True
                    # No dispatch has been enabled. Coordinated rollback is safe
                    # only after every imported mapping has been closed.
                    try:
                        self._release_resources()
                    except Exception:
                        self._pin()
                        logger.exception("PCIe paired initialization rollback could not retire all peer mappings")
                    raise
            self.disabled = False
            logger.info_once(
                "Initialized PCIE_PULL paired all-reduce: TP4, BF16, hidden_size=%d, "
                "capacity=%d elements, blocks=%d, threads=%d.",
                self.HIDDEN_SIZE,
                self.CAPACITY,
                self.BLOCKS,
                self.THREADS,
            )

    def _drain_bound_stream(self) -> None:
        if self._bound_stream is None:
            raise RuntimeError("PCIe paired communicator has no bound stream")
        drained = torch.cuda.Event()
        drained.record(self._bound_stream)
        deadline = time.monotonic() + 3 * self.TIMEOUT_NS / 1_000_000_000 + 0.5
        while not drained.query():
            if time.monotonic() >= deadline:
                raise RuntimeError("PCIe paired lane did not quiesce; retaining its workspace")
            time.sleep(0.001)

    def validate_startup(self) -> None:
        with self._execution(), gpu_sync_allowed(), torch.accelerator.device_index(self.device.index):

            def status():
                if self._bound_stream != current_stream():
                    raise RuntimeError("PCIe paired startup requires the bound main stream")
                self._drain_bound_stream()
                with current_platform.stream(self._bound_stream):
                    code = self._native.local_status(self._ptr)
                if code:
                    raise RuntimeError(f"PCIe paired warmup failed with code {code}")

            self._stage("startup-status", status)

    def should_use(self, inp: torch.Tensor) -> bool:
        # Never consult sticky GPU status or host failure state here: a fault
        # must not cause one rank to switch to a different collective backend.
        return bool(
            self.initialized
            and inp.device == self.device
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and inp.ndim == 2
            and 1 <= inp.shape[0] <= self.MAX_ROWS
            and inp.shape[1] in self.WIDTHS
            and 0 < inp.numel() <= self.CAPACITY
        )

    def should_use_gather(self, inp: torch.Tensor) -> bool:
        # Rank-agreed like should_use: never consult fault or failure state.
        return bool(
            self.initialized
            and inp.device == self.device
            and inp.is_contiguous()
            and inp.ndim >= 1
            and 0 < inp.numel() * inp.element_size() <= self.GATHER_BYTES
        )

    @contextmanager
    def _execution(self):
        if not self._lock.acquire(blocking=False):
            self._failed = True
            self._pin()
            raise RuntimeError("concurrent PCIe paired communicator use is unsupported")
        try:
            if not self.initialized or self._failed:
                raise RuntimeError("PCIe paired communicator is uninitialized or terminally failed")
            yield
        except BaseException:
            if self._ptr:
                self._failed = True
                self._pin()
            raise
        finally:
            self._lock.release()

    def _retain_graph_owner(self) -> None:
        if not current_platform.is_current_stream_capturing():
            return
        lease = self._capture_lease
        if lease is None or not retain_cuda_graph_capture_resource(lease):
            raise RuntimeError("PCIe paired capture requires the TP capture context and graph resource-owner hook")

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        with self._execution():
            if not self.should_use(inp):
                raise ValueError("tensor does not satisfy the rank-agreed PCIe paired admission policy")
            self._retain_graph_owner()
            out = torch.empty_like(inp)
            self._native.all_reduce(self._ptr, inp, out)
            return out

    def all_gather(self, inp: torch.Tensor) -> torch.Tensor:
        """Concatenate every rank's ``inp`` along dim 0 in rank order."""
        with self._execution():
            if not self.should_use_gather(inp):
                raise ValueError("tensor does not satisfy the rank-agreed PCIe gather admission policy")
            self._retain_graph_owner()
            out = inp.new_empty((self.world_size * inp.shape[0], *inp.shape[1:]))
            self._native.all_gather(self._ptr, inp, out)
            return out

    def close_step(self) -> torch.Tensor:
        """Enqueue the collective output gate and return its immutable snapshot.

        The caller orders its existing async output-copy stream after this
        launch, owns the snapshot until its copy event completes, and checks
        the copied integer before releasing any externally visible output.
        """
        with self._execution():
            if current_platform.is_current_stream_capturing():
                raise RuntimeError("the per-output PCIe paired close snapshot must be allocated outside CUDA capture")
            snapshot = torch.empty(1, dtype=torch.int64, device=self.device)
            self._native.close_step(self._ptr, snapshot)
            return snapshot

    @contextmanager
    def capture(self, *, stream: Any):
        """Quiesce and bind before capture, then restore the original main lane.

        Enter this context on the main stream, outside actual CUDA capture and
        *before* the group enters its capture-stream context. Every TP rank must
        enter/leave in the same order. Graph replay stays on the original lane.
        """
        if not self.initialized:
            yield
            return
        with self._execution(), gpu_sync_allowed():
            main_stream = torch.accelerator.current_stream(self.device)

            def preflight():
                if self._capturing:
                    raise RuntimeError("nested PCIe paired capture contexts are unsupported")
                if main_stream != self._bound_stream:
                    raise RuntimeError("PCIe paired capture must begin on the bound main stream")
                torch.accelerator.synchronize(self.device)

            self._stage("capture drain", preflight)
            self._capturing = True
            lease = _GraphLease(self)
            self._capture_lease = lease
            self._graph_leases.add(lease)
            capture_error = None
            try:
                try:
                    with current_platform.stream(stream):
                        self._stage("capture bind", lambda: self._native.bind_stream(self._ptr))
                        yield
                except BaseException as exc:
                    capture_error = f"{type(exc).__name__}: {exc}"
                # Even an exceptional capture must drain on every rank before
                # another stream can be bound. A failed drain leaves it pinned.
                self._stage("capture completion", lambda: torch.accelerator.synchronize(self.device))
                with current_platform.stream(main_stream):
                    self._stage("main stream bind", lambda: self._native.bind_stream(self._ptr))
                    self._bound_stream = main_stream
                self._exchange("capture result", error=capture_error)
            finally:
                self._capture_lease = None
                self._capturing = False

    def _pin(self) -> None:
        if not self._pinned:
            _UNRETIRED_WORKSPACES.append(self)
            self._pinned = True

    def _release_resources(self) -> None:
        def close_imports():
            for peer, pointer in enumerate(self._peer_ptrs):
                if peer != self.rank and pointer:
                    self._native.close_ipc_handle(pointer)
                    self._peer_ptrs[peer] = 0

        self._stage("peer mapping close", close_imports)
        # Every owner retains its raw allocation until every importer has
        # confirmed cudaIpcCloseMemHandle, including initialization rollback.
        self._barrier()
        if self._ptr:
            self._native.dispose(self._ptr)
            self._ptr = 0
        if self._local_ptr:
            ops.free_shared_buffer(self._local_ptr)
            self._local_ptr = 0
            self._peer_ptrs[self.rank] = 0
        self._state = None
        self._ipc_handles = []
        self.disabled = True
        self._destroyed = True

    def destroy(self) -> None:
        """Retire a healthy session collectively; retain poisoned sessions."""
        if self._destroyed or not (self._ptr or self._local_ptr):
            return
        if self._failed or self._pinned:
            self._pin()
            return
        with self._lock, torch.accelerator.device_index(self.device.index), gpu_sync_allowed():
            if self._failed or self._pinned:
                self._pin()
                return
            try:
                self._drain_bound_stream()
                with current_platform.stream(self._bound_stream):
                    status = self._native.local_status(self._ptr)
                if status:
                    self._failed = True
                    self._pin()
                    logger.warning(
                        "Retaining terminal PCIe paired workspace with status %d until process exit.", status
                    )
                    return

                def retire():
                    gc.collect()
                    if self._capturing or self._graph_leases:
                        raise RuntimeError("release every target/draft CUDA graph before communicator destruction")
                    if self._failed:
                        raise RuntimeError("a failed PCIe paired workspace must be retained until process exit")
                    return 0

                self._stage("retirement", retire)
                self._release_resources()
            except BaseException:
                self._failed = True
                self._pin()
                raise

    def __del__(self) -> None:
        # Never issue collectives, cudaFree, or cudaIpcCloseMemHandle here. In
        # particular, imported pointers are not cudaMalloc-owned allocations.
        if getattr(self, "_local_ptr", 0):
            self._pin()
