# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check native indexer row ownership without launching GPU kernels."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("b12x.attention.dsa_indexer")

import b12x.preparation  # noqa: E402

from aphrodite.models.deepseek_v41.sm120 import attention  # noqa: E402


class IndexingComplete(Exception):
    pass


class StopBeforeAttention:
    def __getitem__(self, mode):
        raise IndexingComplete


@pytest.mark.parametrize("rows", [1, 7, 2047, 2048, 2049, 2674, 4095, 4096])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("kind", ["prefill", "mixed", "decode", "capture", "candidate", "consumer"])
def test_indexer_row_ownership(monkeypatch, rows, rank, kind):
    # Request transitions deliberately do not coincide with TP/chunk boundaries.
    reqs = torch.arange(rows, dtype=torch.int32) // 137
    lengths = torch.arange(rows, dtype=torch.int32) % 991 + 1
    table = torch.arange(int(reqs[-1]) + 1, dtype=torch.int32)[:, None] + 17
    expected = (table[reqs.long(), 0] * 10000 + lengths)[:, None].expand(rows, 512)
    metadata = SimpleNamespace(
        is_decode=kind == "decode",
        max_seq_len=20000,
        req_id_per_token=reqs,
        block_table=table,
        cache_lengths=lengths,
    )
    calls = []
    gathered = []
    capture = kind == "capture"
    layer = 20 if kind == "candidate" else 24 if kind == "consumer" else 2
    bounds = [rows * r // 4 for r in range(5)]
    should_shard = rows >= 2048 and kind not in ("decode", "capture", "candidate")

    def gather(local, *, sizes):
        assert should_shard
        assert sizes == [bounds[r + 1] - bounds[r] for r in range(4)]
        assert torch.equal(local, expected[bounds[rank] : bounds[rank + 1]])
        gathered.append(True)
        return expected.clone()

    group = SimpleNamespace(world_size=4, rank_in_group=rank, all_gatherv=gather)

    class Pages:
        def __getitem__(self, grid):
            def run(req_ids, block_table, out, offset, *args):
                count = grid[0]
                out[:count, 0] = block_table[req_ids[offset : offset + count].long(), 0]
                calls.append((offset, count))

            return run

    def bind(plan, **kwargs):
        return SimpleNamespace(**kwargs)

    def select(binding):
        count = binding.q_mxfp4.shape[0]
        values = binding.page_table[:count, 0] * 10000 + binding.cache_lengths
        binding.output_indices.copy_(values[:, None].expand(count, 512))
        if kind == "consumer":
            offset = calls[-1][0]
            assert torch.equal(binding.candidate_lengths, lengths[offset : offset + count])
            assert torch.equal(binding.candidate_indices, reqs[offset : offset + count, None])

    source = SimpleNamespace(_candidates=reqs[:, None], _candidate_lens=lengths)
    fake = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        compress_ratio=2,
        _owner=lambda: SimpleNamespace(prefix="main"),
        _query_metadata=lambda value: value,
        _attention_plan=lambda mode: None,
        _context={"model.layers.2.attn": source, "model.layers.20.attn": source},
        prefix=f"model.layers.{layer}.attn",
        kv_source_layer_id=2,
        index_source_layer_id=2,
        indexer=SimpleNamespace(k_cache=SimpleNamespace(prefix="index", kv_cache=None)),
        layer_id=layer,
        candidate_source_layer=20,
        indexer_tp_shard_min_rows=2048,
        _index_width=8,
        _index_page=128,
        INDEX_CHUNK=256,
        DECODE_CHUNK=64,
        _index_pages=torch.empty(256, 8, dtype=torch.int32),
        _short_index_shape=None,
        _index_plan=lambda mode, count: None,
        _active=None,
        topk_indices_buffer=torch.full((rows, 512), -1, dtype=torch.int32),
        _attention_workspace_specs=StopBeforeAttention(),
        _candidates=torch.empty(rows, 1, dtype=torch.int32),
        _candidate_lens=torch.empty(rows, dtype=torch.int32),
    )
    if kind == "mixed":
        # The first request has a one-token decode query; the remaining rows
        # extend other requests. The metadata still selects the extend path.
        reqs[0] = int(reqs[-1]) + 1
        table = torch.cat((table, table[-1:] + 1))
        metadata.block_table = table
        expected = (table[reqs.long(), 0] * 10000 + lengths)[:, None].expand(rows, 512)
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=dict.fromkeys(("swa", "main", "index"), metadata)),
    )
    monkeypatch.setattr(attention, "get_tp_group", lambda: group)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    monkeypatch.setattr(
        b12x.preparation, "require_prepared", lambda *args: SimpleNamespace(query=SimpleNamespace(query_rows=rows))
    )
    monkeypatch.setattr(attention, "_pages", Pages())
    monkeypatch.setattr(
        attention, "current_workspace_manager", lambda: SimpleNamespace(get_simultaneous=lambda *args: ())
    )
    monkeypatch.setattr(attention, "retain_cuda_graph_capture_resource", lambda scratch: None)
    monkeypatch.setattr(attention.dsa_indexer, "scratch_specs", lambda *args, **kwargs: ())
    monkeypatch.setattr(attention.dsa_indexer, "bind", bind)
    monkeypatch.setattr(attention.dsa_indexer, "score", lambda binding: None)
    monkeypatch.setattr(attention.dsa_indexer, "select", select)
    query = torch.empty(rows, 1)
    with pytest.raises(IndexingComplete):
        attention.DeepseekV41B12xAttention.forward_mqa(fake, query, None, None, None, index_query=(query, query, query))
    begin, end = (bounds[rank], bounds[rank + 1]) if should_shard else (0, rows)
    assert calls[0][0] == begin
    assert sum(count for _, count in calls) == end - begin
    assert bool(gathered) == should_shard
    assert torch.equal(fake.topk_indices_buffer, expected)
