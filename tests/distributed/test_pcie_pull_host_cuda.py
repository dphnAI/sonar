# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import mmap
import os
import tempfile

import pytest
import torch
import torch.distributed as dist

from aphrodite.platforms import current_platform

from ..utils import multi_process_parallel


def _host_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    del monkeypatch, pp_size
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{distributed_init_port}", rank=rank, world_size=tp_size
    )
    import aphrodite._custom_ops  # noqa: F401

    ops = torch.ops._C_pcie_pull
    capacity = 32 * 5120
    slab_bytes = ops.buffer_size(capacity, 1, 3)
    total_bytes = tp_size * slab_bytes
    paths = [""]
    if rank == 0:
        fd, paths[0] = tempfile.mkstemp(prefix="sonar-host-test-", dir="/dev/shm")
        os.ftruncate(fd, total_bytes)
        os.close(fd)
    dist.broadcast_object_list(paths, src=0)
    fd = os.open(paths[0], os.O_RDWR)
    mapping = mmap.mmap(fd, total_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    if rank == 0:
        mapping[:] = bytes(total_bytes)
    dist.barrier()
    if rank == 0:
        # Every peer holds the mapping and its own descriptor; no later native
        # or capture failure can leave a named shared-memory allocation behind.
        os.unlink(paths[0])
    address = ctypes.addressof(ctypes.c_byte.from_buffer(mapping))
    alias = ops.register_host_buffer(address, total_bytes)
    state = torch.zeros(8, dtype=torch.int64, device=device)
    handle = ops.init(
        [alias + peer * slab_bytes for peer in range(tp_size)], state, rank, capacity, 1, 512, 2_000_000_000, 3
    )
    # The stream must remain usable beyond the old 32-bit packet-tag boundary.
    state[0].fill_((1 << 32) - 4)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    packets = []
    for kind, size in (("reduce", 1), ("gather", 7), ("reduce", 8), ("gather", 16), ("reduce", 32)):
        shape = (size, 5120) if kind == "reduce" else (size,)
        dtype = torch.bfloat16 if kind == "reduce" else torch.uint8
        source = torch.empty(shape, dtype=dtype, device=device)
        target = torch.empty(shape if kind == "reduce" else (tp_size * size,), dtype=dtype, device=device)
        packets.append((kind, source, target, torch.empty(1, dtype=torch.int64, device=device)))
    torch.accelerator.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        ops.bind_stream(handle)
        with torch.cuda.graph(graph, stream=stream):
            for kind, source, target, snapshot in packets:
                if kind == "reduce":
                    ops.all_reduce(handle, source, target)
                else:
                    ops.all_gather(handle, source, target)
                ops.close_step(handle, snapshot)
    dist.barrier()
    ops.bind_stream(handle)

    def inputs(source, peer, nonce):
        index = torch.arange(source.numel(), dtype=torch.int64)
        values = (index * (peer * 2 + 3) + nonce * (peer * 5 + 7) + peer * 31) % 127
        return (values - 63).to(source.dtype).view(source.shape) if source.dtype == torch.bfloat16 else values.byte()

    for nonce in range(12):
        for _, source, _, _ in packets:
            source.copy_(inputs(source, rank, nonce))
        if nonce == 6:
            # A live native context must keep its registration usable after a
            # rejected premature release, not merely throw an exception.
            with pytest.raises(RuntimeError):
                ops.unregister_host_buffer(address)
        graph.replay()
        torch.accelerator.synchronize(device)
        for kind, source, target, snapshot in packets:
            assert snapshot.item() == 0
            peers = [inputs(source, peer, nonce) for peer in range(tp_size)]
            if kind == "reduce":
                reference = torch.zeros_like(peers[0], dtype=torch.float32)
                for peer in peers:
                    reference += peer.float()
                reference = reference.bfloat16()
            else:
                reference = torch.cat(peers)
            assert torch.equal(target.cpu(), reference)
    graph.reset()
    torch.accelerator.synchronize(device)
    dist.barrier()
    assert ops.local_status(handle) == 0
    ops.dispose(handle)
    ops.unregister_host_buffer(address)
    dist.barrier()
    mapping.close()
    os.close(fd)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.cuda.device_count() < 4,
    reason="requires four CUDA GPUs",
)
def test_host_graph_replay_close_and_epoch_width(monkeypatch):
    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() != 120:
        pytest.skip("host transport is qualified on SM120")
    ray = pytest.importorskip("ray")
    worker = ray.remote(num_gpus=1, max_calls=1)(_host_worker)
    multi_process_parallel(monkeypatch, 4, 1, worker)
