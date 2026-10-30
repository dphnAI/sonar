# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel rejection sampling against the full-vocabulary oracle.

Four vocab shards are simulated on one GPU: every rank runs its phase on its
slice of the logits, and torch.stack stands in for the all-gather.
"""

import pytest
import torch

from aphrodite.v1.worker.gpu.sample.vocab_parallel import VocabShard
from aphrodite.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    VocabParallelRejection,
    rejection_sample,
    rejection_sample_vocab_parallel,
)

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for rejection sampler tests", allow_module_level=True)

WORLD_SIZE = 4
NUM_SPECULATIVE_STEPS = 4
MAX_NUM_REQS = 16
# A multiple of the 8192-token stats blocks and the 1024-token resample blocks:
# every shard block is also a full-vocab block, so all reductions, and hence all
# outputs, must match bit for bit.
ALIGNED_SHARD_SIZE = 2 * 8192
# Like the production 129280 / 4 = 32320, shard blocks straddle the full-vocab
# block boundaries, so logsumexp and residual mass sums group differently.
UNALIGNED_SHARD_SIZE = 10000

# Draft tokens per request: varied lengths, including a bonus-only request.
_DRAFT_LENS = (4, 0, 1, 2, 3, 4, 4, 1, 4, 3, 2, 4)
# (request, draft step) pairs holding a -1 placeholder draft token. Request 6 is
# greedy in mixed batches.
_PLACEHOLDERS = ((5, 3), (6, 1), (8, 0))
# Softmax-invariant logit shifts used to measure near-ties.
_SHIFTS = (0.3, -0.7, 1.9)


def _make_inputs(
    vocab_size: int,
    mode: str,
    seed: int,
    integer_logits: bool = False,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Return (rejection_sample kwargs without draft_logits, draft_logits,
    per-request greedy mask)."""
    device = "cuda"
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    num_reqs = len(_DRAFT_LENS)
    logits_per_req = torch.tensor(_DRAFT_LENS, device=device) + 1
    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    cu_num_logits[1:] = torch.cumsum(logits_per_req, dim=0)
    starts = cu_num_logits.tolist()
    num_logits = starts[-1]

    idx_mapping = torch.randperm(MAX_NUM_REQS, generator=generator, device=device)[:num_reqs].to(torch.int32)
    expanded_idx_mapping = idx_mapping.repeat_interleave(logits_per_req)
    expanded_local_pos = torch.cat([torch.arange(n, dtype=torch.int32, device=device) for n in logits_per_req.tolist()])
    first_pos = torch.randint(0, 1 << 20, (num_reqs,), generator=generator, device=device, dtype=torch.int32)
    pos = first_pos.repeat_interleave(logits_per_req) + expanded_local_pos

    greedy = torch.tensor(
        [mode == "greedy" or (mode == "mixed" and j % 3 == 0) for j in range(num_reqs)],
        device=device,
    )
    temperature = torch.empty(MAX_NUM_REQS, device=device).uniform_(0.5, 1.5, generator=generator)
    temperature[idx_mapping[greedy].long()] = 0.0
    seeds = torch.randint(0, 1 << 62, (MAX_NUM_REQS,), generator=generator, device=device, dtype=torch.int64)

    if integer_logits:
        # Exact ties everywhere, including across shard boundaries.
        target_logits = torch.randint(0, 4, (num_logits, vocab_size), generator=generator, device=device).float()
    else:
        target_logits = torch.randn(num_logits, vocab_size, generator=generator, device=device) * 3
    # Tokens removed by target-side sampling params.
    masked = torch.rand(num_logits, vocab_size, generator=generator, device=device) < 0.05
    target_logits.masked_fill_(masked, float("-inf"))

    draft_logits = torch.randn(MAX_NUM_REQS, NUM_SPECULATIVE_STEPS, vocab_size, generator=generator, device=device)
    draft_sampled = torch.randint(0, vocab_size, (num_logits,), generator=generator, device=device)
    for j in range(num_reqs):
        state = int(idx_mapping[j])
        temp = float(temperature[state])
        for i in range(_DRAFT_LENS[j]):
            row = starts[j] + i
            if temp == 0.0:
                # Mostly propose the target argmax so greedy drafts get accepted.
                if float(torch.rand((), generator=generator, device=device)) < 0.75:
                    draft_sampled[row + 1] = target_logits[row].argmax()
                continue
            # A draft distribution close to the target, stored pre-temperature.
            noise = torch.randn(vocab_size, generator=generator, device=device)
            q = target_logits[row].nan_to_num(neginf=-10.0) + 0.7 * noise
            draft_logits[state, i] = q * temp
            draft_sampled[row + 1] = torch.multinomial(torch.softmax(q, dim=-1), 1, generator=generator)[0]
    for j, i in _PLACEHOLDERS:
        draft_sampled[starts[j] + i + 1] = -1

    inputs = dict(
        target_logits=target_logits,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temperature,
        seed=seeds,
    )
    return inputs, draft_logits, greedy


def _simulate(
    inputs: dict,
    draft_logits: torch.Tensor | None,
    shard_size: int,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every rank's phases in lockstep, stacking packets for all-gathers."""
    vocab_size = inputs["target_logits"].shape[1]
    assert shard_size * WORLD_SIZE == vocab_size
    samplers = []
    for rank in range(WORLD_SIZE):
        cols = slice(rank * shard_size, (rank + 1) * shard_size)
        shard = VocabShard(
            rank=rank,
            world_size=WORLD_SIZE,
            start=rank * shard_size,
            size=shard_size,
            vocab_size=vocab_size,
        )
        samplers.append(
            VocabParallelRejection(
                inputs["target_logits"][:, cols].contiguous(),
                None if draft_logits is None else draft_logits[..., cols].contiguous(),
                inputs["draft_sampled"],
                inputs["cu_num_logits"],
                inputs["pos"],
                inputs["idx_mapping"],
                inputs["expanded_idx_mapping"],
                inputs["expanded_local_pos"],
                inputs["temperature"],
                inputs["seed"],
                NUM_SPECULATIVE_STEPS,
                shard,
                **kwargs,
            )
        )

    gathered = torch.stack([s.stats_packet() for s in samplers])
    for s in samplers:
        s.apply_stats(gathered)
    if samplers[0].needs_residual_mass:
        gathered = torch.stack([s.residual_mass_packet() for s in samplers])
        for s in samplers:
            s.apply_residual_mass(gathered)
    for s in samplers:
        s.reject()
    gathered = torch.stack([s.resample_packet() for s in samplers]) if samplers[0].needs_resample else None
    outputs = [s.finish(gathered) for s in samplers]

    # The replicated steps see identical gathered data, so ranks must agree.
    for output in outputs[1:]:
        assert not _mismatched(output, outputs[0]).any()
    return outputs[0]


def _masked_tokens(sampled: torch.Tensor, num_sampled: torch.Tensor) -> torch.Tensor:
    # Slots past num_sampled are scratch.
    valid = torch.arange(sampled.shape[1], device=sampled.device) < num_sampled[:, None]
    return torch.where(valid, sampled, -2)


def _mismatched(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Per-request mask of differing outputs."""
    return (_masked_tokens(*actual) != _masked_tokens(*expected)).any(dim=1) | (actual[1] != expected[1])


def _near_ties(
    inputs: dict,
    draft_logits: torch.Tensor | None,
    expected: tuple[torch.Tensor, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    """Requests whose full-vocab result flips under a softmax-invariant shift.

    Adding a constant to the logits changes no distribution and no random
    draw; it only perturbs rounding, by about as much as summing a logsumexp
    in a different order. A request that flips under it sits at a near-tie.
    """
    unstable = torch.zeros_like(expected[1], dtype=torch.bool)
    for shift in _SHIFTS:
        shifted = dict(inputs, target_logits=inputs["target_logits"] + shift)
        output = rejection_sample(
            **shifted,
            draft_logits=None if draft_logits is None else draft_logits + shift,
            num_speculative_steps=NUM_SPECULATIVE_STEPS,
            **kwargs,
        )
        unstable |= _mismatched(output, expected)
    return unstable


def _check_against_full_vocab(
    inputs: dict,
    draft_logits: torch.Tensor | None,
    greedy: torch.Tensor,
    shard_size: int,
    **kwargs,
) -> None:
    expected = rejection_sample(
        **inputs,
        draft_logits=draft_logits,
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
        **kwargs,
    )
    # The all-greedy hint must not change results.
    for all_greedy in (False, True) if bool(greedy.all()) else (False,):
        actual = _simulate(inputs, draft_logits, shard_size, all_greedy=all_greedy, **kwargs)
        mismatched = _mismatched(actual, expected)
        assert not mismatched[greedy].any(), "greedy requests must match bit for bit"
        if shard_size % 8192 == 0:
            assert not mismatched.any(), "aligned shards must match bit for bit"
        elif mismatched.any():
            # Sampled requests can differ only where the full-vocab path is
            # itself unstable under rounding.
            unstable = _near_ties(inputs, draft_logits, expected, **kwargs)
            unexplained = (mismatched & ~unstable).nonzero().flatten().tolist()
            assert not unexplained, f"requests {unexplained} differ away from near-ties"


@pytest.mark.parametrize("shard_size", [ALIGNED_SHARD_SIZE, UNALIGNED_SHARD_SIZE])
@pytest.mark.parametrize("mode", ["mixed", "random", "greedy"])
@pytest.mark.parametrize("use_block_verification", [False, True])
@pytest.mark.parametrize("has_draft_logits", [True, False])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_matches_full_vocab(
    shard_size: int,
    mode: str,
    use_block_verification: bool,
    has_draft_logits: bool,
    use_fp64: bool,
):
    for seed in range(4):
        inputs, draft_logits, greedy = _make_inputs(shard_size * WORLD_SIZE, mode, seed)
        _check_against_full_vocab(
            inputs,
            draft_logits if has_draft_logits else None,
            greedy,
            shard_size,
            use_fp64=use_fp64,
            use_block_verification=use_block_verification,
        )


@pytest.mark.parametrize("shard_size", [ALIGNED_SHARD_SIZE, UNALIGNED_SHARD_SIZE])
@pytest.mark.parametrize("mode", ["mixed", "greedy"])
@pytest.mark.parametrize("use_block_verification", [False, True])
def test_ties_resolve_to_lowest_token(shard_size: int, mode: str, use_block_verification: bool):
    for seed in range(2):
        inputs, draft_logits, greedy = _make_inputs(shard_size * WORLD_SIZE, mode, seed, integer_logits=True)
        _check_against_full_vocab(
            inputs,
            draft_logits,
            greedy,
            shard_size,
            use_block_verification=use_block_verification,
        )


@pytest.mark.parametrize("mode", ["mixed", "greedy"])
@pytest.mark.parametrize("use_block_verification", [False, True])
@pytest.mark.parametrize("has_draft_logits", [True, False])
def test_entry_point_exchanges(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    use_block_verification: bool,
    has_draft_logits: bool,
):
    """The production entry, on one rank owning the whole vocabulary."""
    exchanges = []

    def all_gather(self, tensor):
        exchanges.append(tensor.shape)
        return tensor.unsqueeze(0).clone()

    monkeypatch.setattr(VocabShard, "all_gather", all_gather)
    vocab_size = WORLD_SIZE * UNALIGNED_SHARD_SIZE
    inputs, draft_logits, greedy = _make_inputs(vocab_size, mode, seed=0)
    draft_logits = draft_logits if has_draft_logits else None
    all_greedy = bool(greedy.all())
    shard = VocabShard(rank=0, world_size=1, start=0, size=vocab_size, vocab_size=vocab_size)

    actual = rejection_sample_vocab_parallel(
        **inputs,
        draft_logits=draft_logits,
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
        shard=shard,
        use_block_verification=use_block_verification,
        all_greedy=all_greedy,
    )
    expected = rejection_sample(
        **inputs,
        draft_logits=draft_logits,
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
        use_block_verification=use_block_verification,
    )
    # A single shard reduces exactly like the full vocabulary.
    assert not _mismatched(actual, expected).any()

    if all_greedy:
        expected_exchanges = 1
    elif use_block_verification and has_draft_logits:
        expected_exchanges = 3
    else:
        expected_exchanges = 2
    assert len(exchanges) == expected_exchanges
