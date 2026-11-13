# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-fair KV-read routing across heterogeneous P/D configurations."""

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sonar.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ReqMeta,
    get_port_offset,
)
from sonar.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)

MODE_DIMS = {"TP8": (1, 8), "DP8EP": (8, 1)}


@dataclass(frozen=True)
class PDConfig:
    name: str
    p_mode: str
    d_mode: str

    @property
    def p_dp(self) -> int:
        return MODE_DIMS[self.p_mode][0]

    @property
    def p_tp(self) -> int:
        return MODE_DIMS[self.p_mode][1]

    @property
    def d_dp(self) -> int:
        return MODE_DIMS[self.d_mode][0]

    @property
    def d_tp(self) -> int:
        return MODE_DIMS[self.d_mode][1]

    @property
    def n_prefill_gpus(self) -> int:
        return self.p_dp * self.p_tp


CONFIGS = [
    PDConfig("1P_TP8:1D_TP8", "TP8", "TP8"),
    PDConfig("2P_TP8:1D_DP8EP", "TP8", "DP8EP"),
    PDConfig("2P_TP8:2D_TP8", "TP8", "TP8"),
    PDConfig("2P_DP8EP:3D_DP8EP", "DP8EP", "DP8EP"),
    PDConfig("2P_DP8EP:4D_TP8", "DP8EP", "TP8"),
]


def make_decode_worker(*, world_size: int, tp_rank: int, dp_rank: int):
    worker = object.__new__(MoRIIOConnectorWorker)
    worker.world_size = world_size
    worker.tp_rank = tp_rank
    worker.dp_rank = dp_rank
    worker.use_mla = True
    worker.moriio_config = SimpleNamespace(backend="rdma")
    return worker


def make_meta(*, p_tp: int, p_dp: int, remote_dp_rank: int):
    return ReqMeta(
        transfer_id="t",
        local_block_ids=[1],
        remote_block_ids=[2],
        remote_host="phost0",
        remote_port=1234,
        remote_handshake_port=6301,
        remote_notify_port=61005,
        remote_engine_id="phost0:6301",
        tp_size=p_tp,
        remote_dp_size=p_dp,
        remote_dp_rank=remote_dp_rank,
    )


def build_decode_workers(config: PDConfig) -> list:
    if config.d_tp > 1:
        return [make_decode_worker(world_size=config.d_tp, tp_rank=rank, dp_rank=0) for rank in range(config.d_tp)]
    return [make_decode_worker(world_size=1, tp_rank=0, dp_rank=rank) for rank in range(config.d_dp)]


def prefill_target_multiset(config: PDConfig, rounds: int) -> Counter:
    workers = build_decode_workers(config)
    hits: Counter = Counter()
    for _ in range(rounds):
        for owner_dp in range(config.p_dp):
            for worker in workers:
                meta = make_meta(
                    p_tp=config.p_tp,
                    p_dp=config.p_dp,
                    remote_dp_rank=owner_dp,
                )
                chosen_tp, _ = worker._resolve_read_source(meta)
                target = get_port_offset(owner_dp, chosen_tp, config.p_tp)
                assert 0 <= target < config.n_prefill_gpus
                hits[target] += 1
    return hits


@pytest.mark.parametrize("config", CONFIGS, ids=[c.name for c in CONFIGS])
def test_kv_read_load_is_hardware_fair(config: PDConfig) -> None:
    hits = prefill_target_multiset(config, rounds=16)
    gpus = set(range(config.n_prefill_gpus))
    assert set(hits) == gpus
    assert max(hits.values()) == min(hits.values())


@pytest.mark.parametrize("config", CONFIGS, ids=[c.name for c in CONFIGS])
def test_flexible_gate_fires_only_for_tp_prefill_dp_decode(
    config: PDConfig,
) -> None:
    flags = set()
    for worker in build_decode_workers(config):
        meta = make_meta(p_tp=config.p_tp, p_dp=config.p_dp, remote_dp_rank=0)
        _, flexible = worker._resolve_read_source(meta)
        flags.add(flexible)
    expected = config.d_tp == 1 and config.p_dp == 1 and config.p_tp > 1
    assert flags == {expected}


def test_symmetric_tp_is_a_bijection() -> None:
    targets = []
    for tp_rank in range(8):
        worker = make_decode_worker(world_size=8, tp_rank=tp_rank, dp_rank=0)
        chosen_tp, flexible = worker._resolve_read_source(make_meta(p_tp=8, p_dp=1, remote_dp_rank=0))
        assert not flexible
        targets.append(chosen_tp)
    assert sorted(targets) == list(range(8))


def test_owner_dp_read_is_faithful_and_covers_every_rank() -> None:
    worker = make_decode_worker(world_size=8, tp_rank=3, dp_rank=0)
    targets = []
    for owner_dp in range(8):
        chosen_tp, flexible = worker._resolve_read_source(make_meta(p_tp=1, p_dp=8, remote_dp_rank=owner_dp))
        assert not flexible
        assert chosen_tp == 0
        targets.append(get_port_offset(owner_dp, chosen_tp, 1))
    assert sorted(targets) == list(range(8))


def test_unknown_remote_tp_size_uses_local_size_for_read() -> None:
    worker = make_decode_worker(world_size=4, tp_rank=2, dp_rank=0)
    worker._read_blocks = MagicMock()
    meta = make_meta(p_tp=0, p_dp=2, remote_dp_rank=1)

    worker._read_blocks_for_req("r", meta)

    kwargs = worker._read_blocks.call_args.kwargs
    assert kwargs["remote_tp_size"] == 4
    assert kwargs["chosen_tp"] == 2
    assert get_port_offset(1, kwargs["chosen_tp"], kwargs["remote_tp_size"]) == 6


def test_flexible_round_robin_is_deterministic_uniform_and_staggered() -> None:
    worker = make_decode_worker(world_size=1, tp_rank=0, dp_rank=0)
    sequence = [worker._next_flex_tp_rank(8) for _ in range(64)]
    assert sequence[:8] == list(range(8))
    assert Counter(sequence) == Counter({rank: 8 for rank in range(8)})
    first_pick = [make_decode_worker(world_size=1, tp_rank=0, dp_rank=rank)._next_flex_tp_rank(8) for rank in range(8)]
    assert sorted(first_pick) == list(range(8))


def test_read_blocks_for_req_threads_chosen_tp() -> None:
    worker = make_decode_worker(world_size=1, tp_rank=0, dp_rank=3)
    worker._read_blocks = MagicMock()
    worker._read_blocks_for_req("r", make_meta(p_tp=8, p_dp=1, remote_dp_rank=0))
    kwargs = worker._read_blocks.call_args.kwargs
    assert kwargs["flexible"] is True
    assert kwargs["chosen_tp"] == 3
    assert get_port_offset(0, kwargs["chosen_tp"], 8) == 3
