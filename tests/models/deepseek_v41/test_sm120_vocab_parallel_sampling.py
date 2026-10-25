# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""4-shard single-GPU simulation of vocab-parallel DSpark draft sampling."""

from types import SimpleNamespace

import pytest
import torch

from aphrodite.models.deepseek_v41.sm120.sampling import (
    combine_rounded_sum_partials,
    rounded_sum_shard_partial,
    sample_rounded_sum,
)
from aphrodite.v1.worker.gpu.sample.vocab_parallel import VocabShard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

WORLD = 4
# Not a multiple of the 1024 sampling block, so shard and full block
# boundaries differ.
SHARD = 4100
VOCAB = WORLD * SHARD
STEPS = 3
NUM_REQS = 6


def _shards():
    return [VocabShard(r, WORLD, r * SHARD, SHARD, VOCAB) for r in range(WORLD)]


def _inputs(dtype, seed, greedy_mask):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(greedy_mask)
    base = (torch.randn(batch, VOCAB, generator=gen, device="cuda") * 3).to(dtype)
    bias = torch.randn(batch, VOCAB, generator=gen, device="cuda").to(dtype)
    # Last row maps to a padded (-1) request: no cache write.
    idx_map = torch.arange(batch, device="cuda", dtype=torch.int32) % NUM_REQS
    idx_map[-1] = -1
    temperature = torch.rand(NUM_REQS, generator=gen, device="cuda") + 0.5
    temperature[torch.tensor(greedy_mask, device="cuda")] = 0.0
    seeds = torch.randint(0, 2**31, (NUM_REQS,), generator=gen, device="cuda", dtype=torch.int64)
    positions = torch.randint(0, 4096, (batch,), generator=gen, device="cuda", dtype=torch.int64)
    return base, bias, idx_map, temperature, seeds, positions


def _run(base, bias, idx_map, temperature, seeds, positions, column, use_fp64):
    dtype = base.dtype
    full_cache = torch.full((NUM_REQS, STEPS, VOCAB), float("-inf"), dtype=dtype, device="cuda")
    expected = sample_rounded_sum(
        base, bias, idx_map, temperature, seeds, positions, full_cache, column, use_fp64=use_fp64
    )
    packets = []
    caches = []
    for shard in _shards():
        cache = torch.full((NUM_REQS, STEPS, SHARD), float("-inf"), dtype=dtype, device="cuda")
        cols = slice(shard.start, shard.end)
        packets.append(
            rounded_sum_shard_partial(
                base[:, cols].contiguous(),
                bias[:, cols].contiguous(),
                idx_map,
                temperature,
                seeds,
                positions,
                cache,
                column,
                shard,
                use_fp64=use_fp64,
            )
        )
        caches.append(cache)
    actual = combine_rounded_sum_partials(torch.stack(packets))
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for shard, cache in zip(_shards(), caches):
        assert torch.equal(cache, full_cache[..., shard.start : shard.end])
    return expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("greedy", [(True,) * 6, (False,) * 6, (True, False, True, False, False, True)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_full_vocab(dtype, use_fp64, greedy, seed):
    base, bias, idx_map, temperature, seeds, positions = _inputs(dtype, seed, greedy)
    for step in range(STEPS):
        column = torch.tensor(step, dtype=torch.int32, device="cuda")
        _run(base, bias, idx_map, temperature, seeds, positions, column, use_fp64)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_ties_across_shards_pick_lowest_global_id(dtype, use_fp64):
    batch = 4
    base = torch.zeros(batch, VOCAB, dtype=dtype, device="cuda")
    bias = torch.zeros(batch, VOCAB, dtype=dtype, device="cuda")
    # Row 0: tie at the last column of shard 0 and first of shard 1.
    base[0, SHARD - 1] = 8.0
    base[0, SHARD] = 8.0
    # Row 1: tie across shards 1 and 3; the sum (base + bias) creates it.
    base[1, 2 * SHARD - 5] = 6.0
    bias[1, 2 * SHARD - 5] = 2.0
    base[1, 3 * SHARD + 7] = 8.0
    # Row 2: tie in every shard at the same local offset.
    base[2, torch.arange(WORLD, device="cuda") * SHARD + 11] = 4.0
    # Row 3: all logits equal -> token 0.
    idx_map = torch.arange(batch, device="cuda", dtype=torch.int32)
    temperature = torch.zeros(NUM_REQS, device="cuda")
    seeds = torch.arange(NUM_REQS, device="cuda", dtype=torch.int64)
    positions = torch.zeros(batch, device="cuda", dtype=torch.int64)
    column = torch.tensor(1, dtype=torch.int32, device="cuda")
    ids = _run(base, bias, idx_map, temperature, seeds, positions, column, use_fp64)
    assert ids.tolist() == [SHARD - 1, 2 * SHARD - 5, 11, 0]


def _config(*, batch_sharded=False, watermark=None, return_sampling_mask=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(enable_batch_sharded_sampling=batch_sharded),
        watermark_config=watermark,
        model_config=SimpleNamespace(return_sampling_mask=return_sampling_mask),
    )


def _speculator(**overrides):
    from aphrodite.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.draft_logits = torch.full((2, STEPS, VOCAB), float("-inf"))
    spec._d2t_scatter_index = None
    spec.draft_watermarker = None
    spec.use_acceptance_estimator = False
    spec._draft_topk = None
    spec.enable_adaptive_verification = False
    spec.speculative_config = SimpleNamespace(rejection_sample_method="block")
    spec.max_num_reqs = 2
    spec.num_speculative_steps = STEPS
    spec.device = torch.device("cpu")
    spec.aphrodite_config = _config()
    spec.draft_logits_spec = lambda config: (torch.float32, float("-inf"))
    for name, value in overrides.items():
        setattr(spec, name, value)
    return spec


def _model(shard):
    return SimpleNamespace(draft_vocab_shard=lambda: shard, sample_draft_with_bias_vocab_parallel=lambda *a, **k: None)


def test_speculator_enables_vocab_parallel_draft():
    shard = _shards()[2]
    spec = _speculator()
    spec._maybe_enable_vocab_parallel_draft(_model(shard))
    assert spec.draft_vocab_shard == shard
    assert spec.draft_logits.shape == (2, STEPS, SHARD)
    assert spec.draft_logits.dtype == torch.float32
    assert torch.all(spec.draft_logits == float("-inf"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"draft_logits": None},
        {"_d2t_scatter_index": torch.arange(4)},
        {"draft_watermarker": object()},
        {"use_acceptance_estimator": True},
        {"_draft_topk": 8},
        {"aphrodite_config": _config(batch_sharded=True)},
        {"aphrodite_config": _config(watermark=object())},
        {"aphrodite_config": _config(return_sampling_mask=True)},
        {"enable_adaptive_verification": True},
        {"speculative_config": SimpleNamespace(rejection_sample_method="synthetic")},
    ],
)
def test_speculator_keeps_gathered_path(overrides):
    spec = _speculator(**overrides)
    before = spec.draft_logits
    spec._maybe_enable_vocab_parallel_draft(_model(_shards()[0]))
    assert spec.draft_vocab_shard is None
    assert spec.draft_logits is before


def test_speculator_needs_model_shard():
    spec = _speculator()
    spec._maybe_enable_vocab_parallel_draft(_model(None))
    assert spec.draft_vocab_shard is None
    spec._maybe_enable_vocab_parallel_draft(SimpleNamespace())
    assert spec.draft_vocab_shard is None
