# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sonar.platforms import current_platform
from sonar.sampling_params import SamplingParams
from sonar.v1.sample.ops import SamplingOps
from sonar.v1.worker.gpu.sample.fork import GPUForkSamplingState


def make_state(prompts, outputs):
    req_ids = [f"req{i}" for i in range(len(prompts))]
    tokens = torch.full((len(prompts), 64), -1, dtype=torch.int32)
    for row, (prompt, output) in enumerate(zip(prompts, outputs)):
        tokens[row, : len(prompt) + len(output)] = torch.tensor(prompt + output)
    requests = SimpleNamespace(
        device=torch.device("cpu"),
        vocab_size=16,
        index_to_req_id=dict(enumerate(req_ids)),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        prompt_len=SimpleNamespace(np=np.array(list(map(len, prompts)))),
        prefill_len=SimpleNamespace(np=np.array(list(map(len, prompts)))),
        all_token_ids=SimpleNamespace(gpu=tokens),
    )
    state = GPUForkSamplingState(requests)
    for slot in range(len(prompts)):
        state.add_request(slot, dry_params(), seed=0)
    return state


def dry_params():
    return SamplingParams(temperature=1, dry_multiplier=5, dry_allowed_length=1, dry_sequence_breaker_ids=[])


def apply_history(state, slots, lengths):
    tokens = state.req_states.all_token_ids.gpu
    context = SimpleNamespace(
        expanded_idx_mapping=torch.tensor(slots),
        pos=torch.tensor(lengths) - 1,
        input_ids=torch.tensor([tokens[slot, length - 1] for slot, length in zip(slots, lengths)]),
    )
    return state.apply(torch.zeros((len(slots), 16)), context)


def assert_native_matches_reference(state, slots, lengths):
    current_platform.import_kernels()
    if not hasattr(torch.ops._C, "dry_scan_penalties"):
        pytest.skip("Native DRY scanner is not built on this platform")
    with patch("sonar.v1.sample.ops._compute_dry_penalties", side_effect=AssertionError("Python fallback")):
        native = apply_history(state, slots, lengths)
    for row, slot in enumerate(slots):
        metadata = state.requests[slot].metadata
        with patch.object(metadata, "token_history_ids_cpu", None):
            reference = SamplingOps().apply_dry(torch.zeros((1, 16)), metadata)
        torch.testing.assert_close(native[row : row + 1], reference, rtol=0, atol=0)
    return native


def test_dry_history_grows_without_batch_changes():
    state = make_state([[1, 2]], [[]])
    metadata = state.requests[0].metadata
    apply_history(state, [0], [2])
    assert metadata.token_history_ids_cpu.tolist() == [[1, 2]]
    for end, token in enumerate([1, 2, 1, 2], start=2):
        state.req_states.all_token_ids.gpu[0, end] = token
        result = assert_native_matches_reference(state, [0], [end + 1])
        assert state.requests[0].metadata is metadata
        assert metadata.token_history_ids_cpu.shape == (1, end + 1)
    assert result[0, 1] < 0


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_dry_reordering_and_partial_acceptance(accepted):
    state = make_state([[1, 2], [3, 4]], [[1, 2, 1], [3, 4, 3]])
    tokens = state.req_states.all_token_ids.gpu.clone()
    # Rejected/optimistic suffixes remain in storage, but the current positions
    # bound the history. Batch order is independent of the persistent slot.
    assert_native_matches_reference(state, [1, 0], [2 + accepted] * 2)
    for slot, pair in enumerate(([1, 2], [3, 4])):
        metadata = state.requests[slot].metadata
        assert metadata.token_history_lens_cpu.tolist() == [2 + accepted]
        assert metadata.token_history_ids_cpu.tolist() == [pair + (pair + pair[:1])[:accepted]]
    assert torch.equal(state.req_states.all_token_ids.gpu, tokens)


def test_dry_discarded_outputs_do_not_enter_history():
    state = make_state([[1, 2]], [[1, 2, 9, 9]])
    result = assert_native_matches_reference(state, [0], [4])
    assert state.requests[0].metadata.token_history_ids_cpu.tolist() == [[1, 2, 1, 2]]
    assert result[0, 1] == -8.75
    assert state.req_states.all_token_ids.gpu[0, 4:].tolist() == [9, 9] + [-1] * 58


def test_dry_history_skipped_when_disabled():
    state = make_state([[1, 2]], [[]])
    with patch.object(state, "_token_history", side_effect=AssertionError("Unnecessary history work")):
        assert not state.add_request(0, SamplingParams(), seed=0)
        assert not state.any(np.array([0]))


def test_dry_history_survives_slot_reuse_and_readmission():
    state = make_state([[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]])
    requests = state.req_states
    state.remove_request(1)
    state.finish_request("req1")
    requests.index_to_req_id[1] = "new"
    del requests.req_id_to_index["req1"]
    requests.req_id_to_index["new"] = 1
    requests.all_token_ids.gpu[1, :4] = torch.tensor([7, 8, 7, 8])
    state.add_request(1, dry_params(), seed=0)
    state.remove_request(2)
    requests.prefill_len.np[2] = 4
    state.add_request(2, dry_params(), seed=0)
    assert_native_matches_reference(state, [0, 2, 1], [4, 4, 4])
    assert [state.requests[i].metadata.token_history_ids_cpu.tolist()[0] for i in [0, 2, 1]] == [
        [1, 2, 1, 2],
        [5, 6, 5, 6],
        [7, 8, 7, 8],
    ]
    assert not state.retained


@pytest.mark.parametrize("speculative", [False, True])
def test_dry_refresh_uses_committed_tokens_before_each_sampling_call(speculative):
    state = make_state([[1, 2]], [[1]])
    if speculative:
        # Verification extends a call-local prefix, never the committed buffer.
        context = SimpleNamespace(
            expanded_idx_mapping=torch.tensor([0, 0]),
            pos=torch.tensor([2, 3]),
            input_ids=torch.tensor([1, 9]),
        )
        state.apply(torch.zeros((2, 16)), context)
        assert state.requests[0].metadata.token_history_ids_cpu.tolist() == [[1, 2, 1, 9]]
        assert state.req_states.all_token_ids.gpu[0, 3] == -1
    else:
        apply_history(state, [0], [3])
    # The next decode consumes the accepted output, not the prior draft prefix.
    state.req_states.all_token_ids.gpu[0, 3] = 2
    assert_native_matches_reference(state, [0], [4])
    assert state.requests[0].metadata.token_history_ids_cpu.tolist() == [[1, 2, 1, 2]]
