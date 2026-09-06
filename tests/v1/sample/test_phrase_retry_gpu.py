# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from aphrodite.v1.phrase_guard.v2 import RetryMask


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
def test_v1_retry_in_mixed_speculative_batch(guard_first, greedy):
    from aphrodite.sampling_params import SamplingParams
    from aphrodite.v1.phrase_guard.processor import RETRY_KEY, PhraseRetryProcessor
    from aphrodite.v1.sample.logits_processor import BatchUpdate, LogitsProcessors
    from aphrodite.v1.sample.rejection_sampler import RejectionSampler
    from aphrodite.v1.sample.sampler import Sampler
    from aphrodite.v1.spec_decode.metadata import SpecDecodeMetadata
    from tests.v1.sample.test_rejection_sampler import create_sampling_metadata

    device = torch.device("cuda")
    guard_row = 0 if guard_first else 1
    neighbor_row = 1 - guard_row
    histories = [[], []]
    processor = PhraseRetryProcessor(None, device, False)
    params = SamplingParams(extra_args={RETRY_KEY: (0, [4])})
    processor.update_state(BatchUpdate(2, [], [(guard_row, params, [0], histories[guard_row])], []))
    drafts = [[], [1, 2]] if guard_first else [[1, 2], []]
    metadata = SpecDecodeMetadata.make_dummy(drafts, device)
    metadata.target_logits_indices = torch.tensor([1, 2] if guard_first else [0, 1], device=device)
    metadata.bonus_logits_indices = torch.tensor([0, 3] if guard_first else [2, 3], device=device)
    logits = torch.full((4, 16), -torch.inf, device=device)
    retry_index = 0 if guard_first else 3
    logits[retry_index, 4] = 10.0
    logits[retry_index, 5] = 0.0
    start = 1 if guard_first else 0
    for offset, token in enumerate([1, 2, 3]):
        logits[start + offset, token] = 0.0
    sampling = create_sampling_metadata(
        all_greedy=greedy,
        temperature=torch.full((2,), 0.7, device=device),
        output_token_ids=histories,
        logitsprocs=LogitsProcessors([processor]),
    )
    result = RejectionSampler(Sampler())(metadata, None, logits, sampling).sampled_token_ids.tolist()
    assert result[guard_row] == [5, -1, -1]
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
