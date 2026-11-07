# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer routing fairness and discovery heartbeat lifecycle regressions."""

import contextlib
import ctypes
import importlib.util
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import msgpack
import pytest
import zmq

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def route():
    pytest.importorskip("quart")
    path = Path(__file__).parents[4] / "examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py"
    spec = importlib.util.spec_from_file_location("moriio_proxy_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module.flat_interleaved_dp_route
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("n,dp", [(1, 1), (2, 1), (4, 1), (1, 8), (2, 8), (3, 8)])
def test_all_instance_and_dp_slots_receive_equal_request_shares(route, n, dp):
    instances = [{"dp_size": dp, "tp_size": 8 // dp} for _ in range(n)]
    hits = Counter(route(number, instances) for number in range(1, n * dp * 5 + 1))
    expected = {(instance, None if dp == 1 else rank) for instance in range(n) for rank in range(dp)}
    assert set(hits) == expected
    assert set(hits.values()) == {5}


def test_consecutive_requests_alternate_instances(route):
    instances = [{"dp_size": 8, "tp_size": 1} for _ in range(3)]
    assert [route(number, instances)[0] for number in range(1, 4)] == [0, 1, 2]


@pytest.mark.skipif(sys.platform != "linux", reason="GIL probe uses libc.usleep")
def test_discovery_heartbeat_progresses_while_worker_holds_gil():
    from sonar.distributed.kv_transfer.kv_connector.v1.moriio.moriio_heartbeat import MoRIIOHeartbeat

    payload = {
        "type": "P",
        "http_address": "127.0.0.1:8000",
        "zmq_address": "host:127.0.0.1,handshake:6301,notify:61005",
        "dp_size": 1,
        "tp_size": 8,
        "transfer_mode": "READ",
    }
    with zmq.Context() as context, context.socket(zmq.ROUTER) as socket:
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        heartbeat = MoRIIOHeartbeat(f"tcp://127.0.0.1:{port}", payload, 0.02)
        try:
            assert socket.poll(5000), "Discovery did not register"
            assert msgpack.unpackb(socket.recv_multipart()[1]) == payload
            ctypes.PyDLL(None).usleep(300_000)
            received = 0
            while socket.poll(0):
                assert msgpack.unpackb(socket.recv_multipart()[1]) == payload
                received += 1
            assert received >= 3, "Discovery stalled while the worker held the GIL"
        finally:
            heartbeat.shutdown()
            heartbeat.shutdown()
        while socket.poll(0):
            socket.recv_multipart()
        assert not socket.poll(100), "Heartbeats continued after shutdown"


@pytest.mark.skipif(sys.platform != "linux", reason="Checks child exit via procfs")
def test_discovery_heartbeat_exits_when_worker_dies():
    from sonar.distributed.kv_transfer.kv_connector.v1.moriio import moriio_heartbeat

    owner_code = """
import runpy
import sys
Heartbeat = runpy.run_path(sys.argv[1])["MoRIIOHeartbeat"]
heartbeat = Heartbeat(sys.argv[2], {"type": "D"}, 0.02)
print(heartbeat._process.pid, flush=True)
sys.stdin.read()
"""
    with zmq.Context() as context, context.socket(zmq.ROUTER) as socket:
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        owner = subprocess.Popen(
            [sys.executable, "-c", owner_code, moriio_heartbeat.__file__, f"tcp://127.0.0.1:{port}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        child_pid = None
        child_exited = False
        try:
            assert socket.poll(5000), "Discovery did not register"
            assert owner.stdout is not None
            child_pid = int(owner.stdout.readline())
            owner.kill()
            owner.wait(timeout=5)
            stat = Path(f"/proc/{child_pid}/stat")
            deadline = time.monotonic() + 5
            while stat.exists():
                try:
                    if stat.read_text().split(") ")[1].startswith("Z "):
                        break
                except FileNotFoundError:
                    break
                assert time.monotonic() < deadline, "Orphan heartbeat survived worker"
                time.sleep(0.01)
            child_exited = True
        finally:
            if owner.poll() is None:
                owner.kill()
            owner.communicate(timeout=5)
            if child_pid is not None and not child_exited:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child_pid, 9)


def test_discovery_heartbeat_recovers_after_router_outage(capfd):
    from sonar.distributed.kv_transfer.kv_connector.v1.moriio.moriio_heartbeat import MoRIIOHeartbeat

    payload = {"type": "P", "http_address": "127.0.0.1:8000"}
    with zmq.Context() as context:
        with context.socket(zmq.ROUTER) as socket:
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            endpoint = f"tcp://127.0.0.1:{port}"
            heartbeat = MoRIIOHeartbeat(endpoint, payload, 0.001, max_retries=1)
            try:
                assert socket.poll(5000), "Discovery did not register"
                assert msgpack.unpackb(socket.recv_multipart()[1]) == payload
            except BaseException:
                heartbeat.shutdown()
                raise
        try:
            deadline = time.monotonic() + 10
            errors = ""
            while "heartbeat send failed" not in errors:
                errors += capfd.readouterr().err
                assert time.monotonic() < deadline, "No send timeout during outage"
                time.sleep(0.01)
            with context.socket(zmq.ROUTER) as socket:
                socket.bind(endpoint)
                assert socket.poll(5000), "Discovery did not recover after outage"
                assert msgpack.unpackb(socket.recv_multipart()[1]) == payload
                assert heartbeat._process.poll() is None
        finally:
            heartbeat.shutdown()
