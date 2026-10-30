# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and graph regressions for the SM120 sparse-attention indexer."""

import pytest
import torch

from aphrodite.model_executor.kernels.attention.dsa.sm120_indexer import (
    sm120_fp8_mqa_logits,
    sm120_fp8_paged_mqa_logits,
)
from aphrodite.platforms import current_platform

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64


def _sm120_available() -> bool:
    if not current_platform.is_cuda():
        return False
    try:
        return current_platform.is_device_capability_family(120)
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 GPU")


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (scores.mul(scales[None, None, :]).relu() * weights[:, :, None]).sum(1)


def test_sm120_fp8_mqa_logits() -> None:
    torch.manual_seed(0)
    num_rows, num_tokens = 5, 193
    q = torch.randn((num_rows, NUM_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn((num_tokens, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scales = torch.rand(num_tokens, device="cuda") * 0.02
    weights = torch.randn((num_rows, NUM_HEADS), device="cuda")
    starts = torch.tensor([0, 3, 61, 100, 192], dtype=torch.int32, device="cuda")
    ends = torch.tensor([193, 129, 162, 193, 193], dtype=torch.int32, device="cuda")

    actual = sm120_fp8_mqa_logits(q, k, scales, weights, starts, ends)
    expected = _reference(q, k, scales, weights)
    positions = torch.arange(num_tokens, device="cuda")[None, :]
    valid = (positions >= starts[:, None]) & (positions < ends[:, None])
    expected.masked_fill_(~valid, -torch.inf)

    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-2, atol=2e-2)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


def test_sm120_ragged_logits_graph_updates_bounds() -> None:
    torch.manual_seed(3)
    rows, heads, num_tokens = 4, 32, 20000
    q = torch.randn((rows, heads, HEAD_DIM + 16), device="cuda").to(torch.float8_e4m3fn)[..., :HEAD_DIM]
    k = torch.randn((num_tokens, HEAD_DIM + 32), device="cuda").to(torch.float8_e4m3fn)[:, :HEAD_DIM]
    scales = torch.rand(num_tokens, device="cuda") * 0.02
    weights = torch.randn((rows, heads + 8), device="cuda")[:, :heads]
    starts = torch.tensor([0, 1, 8193, 19999], dtype=torch.int32, device="cuda")
    ends = torch.tensor([17001, 65, 17500, 20000], dtype=torch.int32, device="cuda")
    reference = _reference(q, k, scales, weights)
    sm120_fp8_mqa_logits(q, k, scales, weights, starts, ends)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sm120_fp8_mqa_logits(q, k, scales, weights, starts, ends)

    for begin, finish in (
        ([0, 1, 8193, 19999], [17001, 65, 17500, 20000]),
        ([17000, 63, 0, 20000], [17001, 64, 0, 20000]),
        ([0, 127, 8191, 65], [20000, 129, 20000, 130]),
    ):
        starts.copy_(torch.tensor(begin, dtype=torch.int32, device="cuda"))
        ends.copy_(torch.tensor(finish, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        positions = torch.arange(num_tokens, device="cuda")[None, :]
        valid = (positions >= starts[:, None]) & (positions < ends[:, None])
        torch.testing.assert_close(output[valid], reference[valid], rtol=2e-2, atol=2e-2)
        assert torch.equal(torch.isneginf(output), ~valid)


@pytest.mark.parametrize("next_n", [1, 2, 7])
@pytest.mark.parametrize(
    ("storage_offset", "page_padding"),
    [(0, 0), (4096, 512)],
    ids=["standalone", "shared-storage"],
)
def test_sm120_fp8_paged_mqa_logits(next_n: int, storage_offset: int, page_padding: int) -> None:
    torch.manual_seed(1)
    batch_size, num_pages = 2, 5
    max_model_len = 192
    q = torch.randn(
        (batch_size, next_n, NUM_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    keys = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scales = torch.rand((num_pages, PAGE_SIZE), device="cuda") * 0.02
    page_size_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    page_stride = page_size_bytes + page_padding
    storage = torch.full(
        (storage_offset + num_pages * page_stride,),
        0xFF,
        dtype=torch.uint8,
        device="cuda",
    )
    cache = torch.as_strided(
        storage,
        (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
        (page_stride, HEAD_DIM + 4, HEAD_DIM + 4, 1),
        storage_offset=storage_offset,
    )
    # Match indexer_k_quant_and_cache's planar page layout: all key rows,
    # followed by all per-token scales.
    page_bytes = cache.as_strided((num_pages, page_size_bytes), (page_stride, 1))
    page_bytes[:, : PAGE_SIZE * HEAD_DIM].copy_(keys.view(torch.uint8).reshape(num_pages, -1))
    page_bytes[:, PAGE_SIZE * HEAD_DIM :].copy_(scales.view(torch.uint8).reshape(num_pages, -1))
    weights = torch.randn((batch_size * next_n, NUM_HEADS), device="cuda")
    context_lens = torch.tensor(
        [[129 + token for token in range(next_n)], [70 + token for token in range(next_n)]],
        dtype=torch.int32,
        device="cuda",
    )
    block_tables = torch.tensor([[2, 0, 4], [3, 1, 0]], dtype=torch.int32, device="cuda")

    actual = sm120_fp8_paged_mqa_logits(q, cache, weights, context_lens, block_tables, max_model_len)
    expected = torch.full_like(actual, -torch.inf)
    for batch in range(batch_size):
        pages = block_tables[batch].long()
        logical_k = keys[pages].reshape(-1, HEAD_DIM)
        logical_scales = scales[pages].reshape(-1)
        for token in range(next_n):
            row = batch * next_n + token
            length = int(context_lens[batch, token])
            expected[row, :length] = _reference(
                q[batch, token].unsqueeze(0),
                logical_k[:length],
                logical_scales[:length],
                weights[row].unsqueeze(0),
            )[0]

    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=2e-2, atol=2e-2)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


@pytest.mark.parametrize("backend", ["triton", "deep_gemm"])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("next_n", [1, 2, 4, 6, 8])
@torch.inference_mode()
def test_sm120_paged_logits_graph_updates_inputs(backend: str, page_size: int, next_n: int) -> None:
    from aphrodite.model_executor.kernels.attention.dsa.sm120_indexer import use_sm120_dsa_indexer
    from aphrodite.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
    from aphrodite.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_num_sms, get_paged_mqa_logits_metadata

    if backend == "deep_gemm" and use_sm120_dsa_indexer(page_size):
        pytest.skip(f"installed DeepGEMM lacks native SM120 indexers for {page_size}-row pages")
    torch.manual_seed(17)
    batch, num_heads, max_model_len, topk = 2, 32, 32768, 512
    rows = batch * next_n
    table_width = max_model_len // page_size
    num_pages = batch * table_width
    q = torch.randint(-2, 3, (batch, next_n, num_heads, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randint(-2, 3, (num_pages, page_size, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    scales = 2.0 ** torch.randint(-10, -7, (num_pages, page_size), device="cuda").float()
    page_bytes = page_size * (HEAD_DIM + 4)
    offset, stride = 4096, page_bytes + 512
    storage = torch.full((offset + num_pages * stride,), 0xA5, dtype=torch.uint8, device="cuda")
    cache = storage.as_strided((num_pages, page_size, 1, HEAD_DIM + 4), (stride, HEAD_DIM + 4, HEAD_DIM + 4, 1), offset)
    packed_pages = cache.as_strided((num_pages, page_bytes), (stride, 1))
    weights = torch.randint(0, 4, (rows, num_heads), device="cuda").float()
    tables = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(batch, table_width)
    lengths = torch.empty((batch, next_n), dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    workspace = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")
    num_sms = get_num_sms() if backend == "deep_gemm" else 0

    def update(finals: tuple[int, int]) -> None:
        q.copy_(torch.randint(-2, 3, q.shape, device="cuda").to(q.dtype))
        weights.copy_(torch.randint(0, 4, weights.shape, device="cuda").float())
        keys.copy_(keys.flip(1))
        packed_pages[:, : page_size * HEAD_DIM].copy_(keys.view(torch.uint8).reshape(num_pages, -1))
        packed_pages[:, page_size * HEAD_DIM :].copy_(scales.view(torch.uint8).reshape(num_pages, -1))
        tables.copy_(tables.flip(1))
        for request, final in enumerate(finals):
            lengths[request].copy_(
                (final - next_n + 1 + torch.arange(next_n, dtype=torch.int32, device="cuda")).clamp_min_(0)
            )

    def run() -> torch.Tensor:
        if backend == "triton":
            logits = sm120_fp8_paged_mqa_logits(q, cache, weights, lengths, tables, max_model_len)
        else:
            schedule = get_paged_mqa_logits_metadata(lengths, page_size, num_sms)
            logits = fp8_fp4_paged_mqa_logits((q, None), cache, weights, lengths, tables, schedule, max_model_len)
        torch.ops._C.persistent_topk(logits, lengths, indices, workspace, topk, max_model_len)
        return logits

    update((17023, 8193))
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for finals in ((17023, 8193), (65, 0), (0, 17024), (17024, 513)):
        update(finals)
        graph.replay()
        for request in range(batch):
            logical_keys = keys[tables[request].long()].reshape(-1, HEAD_DIM)
            logical_scales = scales[tables[request].long()].reshape(-1)
            for query in range(next_n):
                row = request * next_n + query
                length = int(lengths[request, query])
                reference = _reference(
                    q[request, query : query + 1],
                    logical_keys[:length],
                    logical_scales[:length],
                    weights[row : row + 1],
                )[0]
                torch.testing.assert_close(actual[row, :length], reference, rtol=0, atol=0)
                # Native logits leave tails undefined; only the fallback promises -inf.
                if backend == "triton":
                    assert bool(torch.isneginf(actual[row, length:]).all())
                selected = indices[row]
                valid = selected[selected >= 0].long()
                count = min(topk, length)
                assert valid.numel() == valid.unique().numel() == count
                assert int((selected == -1).sum()) == topk - count
                assert bool((valid < length).all())
                torch.testing.assert_close(
                    reference[valid].sort(descending=True).values, reference.topk(count).values, rtol=0, atol=0
                )
    graph.reset()


@pytest.mark.parametrize("backend", ["triton", "deep_gemm"])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@torch.inference_mode()
def test_ragged_decode_preserves_query_weights_and_causal_bounds(
    backend: str, page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import aphrodite.model_executor.layers.sparse_attn_indexer as indexer
    from aphrodite.config import CUDAGraphMode
    from aphrodite.model_executor.kernels.attention.dsa.sm120_indexer import use_sm120_dsa_indexer
    from aphrodite.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from aphrodite.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerMetadataBuilder,
    )
    from aphrodite.v1.worker import workspace

    if backend == "deep_gemm" and use_sm120_dsa_indexer(page_size):
        pytest.skip(f"installed DeepGEMM lacks native SM120 indexers for {page_size}-row pages")
    monkeypatch.setattr(
        indexer,
        "use_sm120_dsa_indexer",
        lambda page_size=None: backend == "triton" or use_sm120_dsa_indexer(page_size),
    )
    torch.manual_seed(83)
    rows, heads, width, topk = 6, 32, 2048, 512
    counts, finals = (0, 1, 2, 3), (0, 1024, 1536, 2016)
    num_requests = len(counts)
    padded_rows = num_requests * max(counts)
    pages = num_requests * (width // page_size)
    q = torch.randn((rows, heads, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randn((pages, page_size, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((pages, page_size), device="cuda") * 0.02
    if page_size == 128:
        # Shared planar cache storage with a nonzero layer offset.
        offset, stride = 213120, 230400
        storage = torch.empty((offset + pages * stride,), dtype=torch.uint8, device="cuda")
        cache = storage.as_strided((pages, page_size, HEAD_DIM + 4), (stride, HEAD_DIM + 4, 1), offset)
    else:
        cache = torch.empty((pages, page_size, HEAD_DIM + 4), dtype=torch.uint8, device="cuda")
    page_bytes = cache.view(pages, -1)
    weights = torch.full((rows, heads), 0.02, device="cuda")
    weights[torch.arange(rows, device="cuda"), torch.arange(rows, device="cuda") * 5 % heads] = 1
    lengths = torch.tensor(counts, dtype=torch.int32, device="cuda")
    final_contexts = torch.tensor(finals, dtype=torch.int32, device="cuda")
    tables = torch.randperm(pages, device="cuda").reshape(num_requests, -1).int()
    host_tables = tables.cpu().tolist()
    row = 0
    for request, count in enumerate(counts):
        for query in range(count):
            position = finals[request] - count + query
            page, slot = host_tables[request][position // page_size], position % page_size
            keys[page, slot].copy_((q[row, row * 5 % heads].float() * 8).to(keys.dtype))
            scales[page, slot] = 0.02
            row += 1
    page_bytes[:, : page_size * HEAD_DIM].copy_(keys.view(torch.uint8).reshape(pages, -1))
    page_bytes[:, page_size * HEAD_DIM :].copy_(scales.view(torch.uint8).reshape(pages, -1))
    builder = SimpleNamespace(
        aphrodite_config=SimpleNamespace(speculative_config=None),
        decode_seq_lens_buffer=torch.empty((padded_rows,), dtype=torch.int32, device="cuda"),
        offsets_buffer=torch.arange(3, dtype=torch.int32, device="cuda"),
    )
    contexts, _, _, _, requires_padding = DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors(
        builder,
        seq_lens=final_contexts,
        block_table=tables,
        decode_lens=lengths,
        decode_lens_cpu=lengths.cpu(),
        query_start_loc=torch.tensor([0, 0, 1, 3], dtype=torch.int32, device="cuda"),
        num_decodes=num_requests,
        num_decode_tokens=rows,
        use_native=True,
        next_n=3,
        max_decode_len=3,
    )
    schedule = (
        get_paged_mqa_logits_metadata(contexts, page_size, get_num_sms())
        if not indexer.use_sm120_dsa_indexer(page_size)
        else torch.empty((0, 2), dtype=torch.int32, device="cuda")
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=final_contexts,
        max_seq_len=width,
        slot_mapping=torch.full((rows,), -1, dtype=torch.long, device="cuda"),
        num_decodes=num_requests,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(tables, contexts, lengths, requires_padding, schedule),
    )
    context = SimpleNamespace(attn_metadata={"test.indexer": metadata}, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(indexer, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        workspace,
        "_manager",
        workspace.WorkspaceManager(torch.device("cuda"), num_ubatches=1, num_lanes=1),
    )
    output = torch.empty((padded_rows, topk), dtype=torch.int32, device="cuda")
    indexer.sparse_attn_indexer(
        torch.empty((rows, 1), device="cuda"),
        "test.indexer",
        cache,
        q,
        None,
        None,
        weights,
        128,
        None,
        topk,
        HEAD_DIM,
        width,
        pages * page_size,
        output,
        True,
        False,
        False,
        "",
        topk_backend="persistent",
    )
    row = 0
    for request, count in enumerate(counts):
        logical_k = keys[tables[request].long()].reshape(-1, HEAD_DIM)
        logical_scales = scales[tables[request].long()].reshape(-1)
        for query in range(count):
            length = finals[request] - count + query + 1
            reference = _reference(
                q[row : row + 1], logical_k[:length], logical_scales[:length], weights[row : row + 1]
            )[0]
            indices = output[row].long()
            assert bool(((indices >= 0) & (indices < length)).all())
            assert indices.unique().numel() == topk
            torch.testing.assert_close(
                reference[indices].sort(descending=True).values,
                reference.topk(topk).values,
                rtol=1e-4,
                atol=1e-5,
            )
            row += 1


@pytest.mark.parametrize("next_n", [1, 2, 4, 6, 8])
def test_sm120_candidate_topk_graph_preserves_pool_and_causal_bounds(monkeypatch, next_n: int) -> None:
    from aphrodite.model_executor.kernels.attention.dsa.sm120_candidate_topk import (
        reserve_sm120_candidate_topk,
        sm120_fp8_candidate_topk,
    )
    from aphrodite.v1.worker import workspace

    torch.manual_seed(31)
    batch, heads, page_size, width = 2, 32, 128, 32769
    rows = batch * next_n
    table_width = (width + page_size - 1) // page_size
    pages, stride, offset = batch * table_width, 230400, 213120
    storage = torch.full((offset + pages * stride,), 0xA5, dtype=torch.uint8, device="cuda")
    cache = torch.as_strided(storage, (pages, page_size, HEAD_DIM + 4), (stride, HEAD_DIM + 4, 1), offset)
    keys = torch.randint(-2, 3, (pages, page_size, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    scales = 2.0 ** torch.randint(-10, -7, (pages, page_size), device="cuda").float()
    page_bytes = torch.as_strided(cache, (pages, page_size * (HEAD_DIM + 4)), (stride, 1))
    page_bytes[:, : page_size * HEAD_DIM].copy_(keys.view(torch.uint8).reshape(pages, -1))
    page_bytes[:, page_size * HEAD_DIM :].copy_(scales.view(torch.uint8).reshape(pages, -1))
    saved_storage = storage.clone()
    tables = torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(batch, table_width)
    q = torch.randint(-2, 3, (batch, next_n, heads, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randint(0, 4, (rows, heads), device="cuda").float()
    lengths = torch.empty((batch, next_n), dtype=torch.int32, device="cuda")
    candidates = torch.full((rows + 3, 2048), -1, dtype=torch.int32, device="cuda")
    output = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    manager = workspace.WorkspaceManager(torch.device("cuda"), num_ubatches=1, num_lanes=1)
    monkeypatch.setattr(workspace, "_manager", manager)
    reserve_sm120_candidate_topk(candidates.shape[0])

    def set_bounds(finals, shifted):
        for request, final in enumerate(finals):
            for query in range(next_n):
                row = request * next_n + query
                length = max(0, final - next_n + query + 1)
                lengths[request, query] = length
                blocks = torch.arange(2048, dtype=torch.int32, device="cuda")
                if shifted and length >= 32764:
                    blocks += 1800
                candidates[row].copy_(torch.where(blocks * 8 < length, blocks, -1))

    def run():
        sm120_fp8_candidate_topk(q, cache, weights, lengths, tables, candidates, width, output)

    set_bounds((width, 8193), False)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    manager.lock()
    for finals, shifted in (((width, 8193), False), ((513, 0), False), ((width, 8193), True)):
        set_bounds(finals, shifted)
        q.copy_(torch.randint(-2, 3, q.shape, device="cuda").to(q.dtype))
        weights.copy_(torch.randint(0, 4, weights.shape, device="cuda").float())
        tables.copy_(tables.flip(1))
        graph.replay()
        selected = output.clone()
        run()
        for request in range(batch):
            logical_keys = keys[tables[request].long()].reshape(-1, HEAD_DIM)
            logical_scales = scales[tables[request].long()].reshape(-1)
            for query in range(next_n):
                row = request * next_n + query
                length = int(lengths[request, query])
                blocks = candidates[row][candidates[row] >= 0].long()
                positions = (blocks[:, None] * 8 + torch.arange(8, device="cuda")).flatten()
                positions = positions[positions < length]
                reference = _reference(
                    q[request, query : query + 1],
                    logical_keys[:length],
                    logical_scales[:length],
                    weights[row : row + 1],
                )[0]
                count = min(512, positions.numel())
                expected = reference[positions].topk(count).values
                for actual in (selected[row], output[row]):
                    valid = actual[actual >= 0].long()
                    assert valid.numel() == count and valid.unique().numel() == count
                    assert int((actual == -1).sum()) == 512 - count
                    assert bool(torch.isin(valid, positions).all())
                    torch.testing.assert_close(reference[valid].sort(descending=True).values, expected, rtol=0, atol=0)
        assert torch.equal(storage, saved_storage)
    graph.reset()
