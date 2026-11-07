# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sonar.sampling_params import SamplingParams
from sonar.v1.worker.gpu.sample.fork import GPUForkSamplingState, needs_fork_sampling

pytestmark = pytest.mark.cpu_test


def _request_states():
    return SimpleNamespace(
        device=torch.device("cpu"),
        vocab_size=8,
        index_to_req_id={0: "request"},
        req_id_to_index={"request": 0},
        prompt_len=SimpleNamespace(np=np.array([2, 2])),
        prefill_len=SimpleNamespace(np=np.array([2, 2])),
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0]])),
    )


def _context(slot, positions, input_ids):
    return SimpleNamespace(
        expanded_idx_mapping=torch.tensor([slot] * len(positions)),
        pos=torch.tensor(positions),
        input_ids=torch.tensor(input_ids),
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"dry_multiplier": 1.0},
        {"no_repeat_ngram_size": 2},
        {"dynatemp_min": 0.5, "dynatemp_max": 1.5},
        {"temperature_last": True},
        {"top_a": 0.1},
        {"tfs": 0.9},
        {"eta_cutoff": 0.1},
        {"epsilon_cutoff": 0.1},
        {"typical_p": 0.9},
        {"smoothing_factor": 0.2},
        {"xtc_probability": 0.1},
        {"nsigma": 1.0},
        {"mirostat_mode": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
        {"skew": 1.0},
    ],
)
def test_custom_sampling_parameters_select_fork_path(extra):
    assert needs_fork_sampling(SamplingParams(**extra))


def test_native_sampling_parameters_keep_native_path():
    state = GPUForkSamplingState(_request_states())
    params = SamplingParams(temperature=0.7, top_k=4, top_p=0.9, min_p=0.1, repetition_penalty=1.1)
    assert not state.add_request(0, params, seed=11)
    assert not state.any(np.array([0]))


@pytest.mark.parametrize("accepted_count", [1, 2])
def test_mirostat_commits_only_accepted_tokens_and_restores_on_rewind(accepted_count):
    requests = _request_states()
    state = GPUForkSamplingState(requests)
    params = SamplingParams(mirostat_mode=2, mirostat_tau=2.0, mirostat_eta=0.5)
    assert state.add_request(0, params, seed=11)
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0],
        ]
    )
    surprise = -torch.log_softmax(logits, dim=-1) / np.log(2.0)
    state.apply(logits.clone(), _context(0, [1, 2, 3], [2, 3, 4]))
    assert state.requests[0].checkpoints == {0: 4.0}
    sampled = SimpleNamespace(sampled_token_ids=torch.tensor([[3, 4, 5]]), num_sampled=torch.tensor([accepted_count]))
    batch = SimpleNamespace(idx_mapping_np=np.array([0]))
    state.update(batch, sampled)
    expected = {0: 4.0}
    for index, token in enumerate([3, 4][:accepted_count]):
        expected[index + 1] = expected[index] - params.mirostat_eta * (
            float(surprise[index, token]) - params.mirostat_tau
        )
    assert state.requests[0].checkpoints == pytest.approx(expected)
    state.update(batch, sampled)
    assert state.requests[0].checkpoints == pytest.approx(expected)

    # Preemption may give the request a new slot; rollback discards all but
    # the first accepted output before the request is admitted again.
    state.remove_request(0)
    requests.index_to_req_id = {1: "request"}
    requests.req_id_to_index = {"request": 1}
    requests.prefill_len.np[1] = 3
    assert state.add_request(1, params, seed=99)
    assert state.requests[1].checkpoints == pytest.approx({0: expected[0], 1: expected[1]})
    next_logits = torch.tensor([[0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0]])
    next_surprise = -torch.log_softmax(next_logits, dim=-1) / np.log(2.0)
    support = next_surprise <= expected[1]
    support[0, next_surprise.argmin()] = True
    restored = state.apply(next_logits.clone(), _context(1, [2], [3]))
    assert torch.equal(torch.isfinite(restored), support)

    state.finish_request("request")
    assert not state.any(np.array([1]))
    assert not state.retained
