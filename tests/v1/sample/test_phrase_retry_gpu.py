# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from sonar.v1.phrase_guard.v2 import RetryMask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_retry_mask_only_changes_checkpoint_rows_and_clears_reused_slots():
    mask = RetryMask(3, torch.device("cuda"))
    mask.add_request(2, 12, (3, [4, 7]))
    mask.add_request(0, 10, None)
    mask.apply_staged_writes()
    mapping = torch.tensor([2, 2, 0, 2], device="cuda")
    positions = torch.tensor([14, 15, 14, 16], device="cuda")
    logits = torch.zeros((4, 16), device="cuda")
    mask.apply(logits, mapping, np.array([2, 0]), positions)
    expected = torch.zeros_like(logits)
    expected[0, [4, 7]] = -torch.inf
    torch.testing.assert_close(logits, expected)
    mask.add_request(2, 12, None)
    mask.apply_staged_writes()
    logits.zero_()
    mask.apply(logits, mapping, np.array([2, 0]), positions)
    assert not logits.count_nonzero().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("guard_first", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_retry_in_mixed_speculative_batch(guard_first, greedy):
    from types import SimpleNamespace

    from sonar.sampling_params import SamplingParams
    from sonar.v1.phrase_guard.processor import RETRY_KEY
    from sonar.v1.worker.gpu.sample.sampler import Sampler
    from sonar.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from sonar.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    guard_row = 0 if guard_first else 1
    neighbor_row = 1 - guard_row
    states = RequestState(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=4,
        num_speculative_steps=2,
        vocab_size=16,
        device=device,
    )
    states.prompt_len.np[:] = 1
    sampler = Sampler(SimpleNamespace(reasoning_config=None), 2, 16, device, states)
    for row in range(2):
        sampler.add_request(
            row,
            SamplingParams(
                temperature=0.0 if greedy else 0.7,
                seed=17,
                extra_args={RETRY_KEY: (0, [4])} if row == guard_row else None,
            ),
        )
    sampler.apply_staged_writes()
    rejection = RejectionSampler(
        sampler,
        SimpleNamespace(
            num_speculative_tokens=2,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
            fly_entropy_threshold=0.0,
            fly_entropy_top_k=0,
        ),
        device,
    )
    counts = [1, 3] if guard_first else [3, 1]
    mapping = torch.tensor([row for row, count in enumerate(counts) for _ in range(count)], device=device)
    local_pos = torch.tensor([pos for count in counts for pos in range(count)], device=device)
    cu_num_logits = torch.tensor([0, counts[0], 4], dtype=torch.int32, device=device)
    drafts = torch.tensor([0, 0, 1, 2] if guard_first else [0, 1, 2, 0], device=device)
    idx_mapping_np = np.array([0, 1], dtype=np.int32)
    idx_mapping = torch.from_numpy(idx_mapping_np).to(device)
    logits = torch.full((4, 16), -torch.inf, device=device)
    retry_index = 0 if guard_first else 3
    logits[retry_index, 4] = 10.0
    logits[retry_index, 5] = 0.0
    start = 1 if guard_first else 0
    for offset, token in enumerate([1, 2, 3]):
        logits[start + offset, token] = 0.0
    _, sampled, num_sampled = rejection._verify(
        logits,
        None,
        drafts,
        local_pos,
        cu_num_logits,
        idx_mapping,
        idx_mapping_np,
        mapping,
        local_pos,
        np.array(counts, dtype=np.int32),
    )
    result = sampled.tolist()
    assert num_sampled.tolist()[guard_row] == 1
    assert num_sampled.tolist()[neighbor_row] == 3
    assert result[guard_row][0] == 5
    assert result[neighbor_row] == [1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("vocab_size", [16, 151936, 256128])
def test_exhausted_retry_returns_sentinel_without_changing_other_rows(vocab_size):
    mask = RetryMask(2, torch.device("cuda"))
    mask.add_request(0, 12, (0, [4, 7]))
    mask.add_request(1, 12, (0, [4, 7]))
    mask.apply_staged_writes()
    mapping = torch.tensor([0, 1], device="cuda")
    positions = torch.tensor([11, 11], device="cuda")
    logits = torch.full((2, vocab_size), -torch.inf, device="cuda")
    logits[0, 7] = 2.0
    logits[1, -1] = 3.0
    expected = logits.clone()
    expected[0, 7] = -torch.inf
    expected[0, 4] = 0.0
    mask.apply(logits, mapping, np.array([0, 1]), positions)
    torch.testing.assert_close(logits, expected)
