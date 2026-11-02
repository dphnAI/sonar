# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from aphrodite.platforms import current_platform
from aphrodite.sampling_params import SamplingParams
from aphrodite.v1.sample.ops import SamplingOps
from aphrodite.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from aphrodite.v1.worker.gpu_model_runner import GPUModelRunner


def make_batch(prompts, outputs):
    batch = object.__new__(InputBatch)
    req_ids = [f"req{i}" for i in range(len(prompts))]
    batch._req_ids = list(req_ids)
    batch.req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
    batch.dry_reqs = set(req_ids)
    batch.req_output_token_ids = outputs
    batch.num_prompt_tokens = np.array(list(map(len, prompts)), dtype=np.int32)
    batch.token_ids_cpu_tensor = torch.full((len(prompts), 64), -1, dtype=torch.int32)
    batch.token_ids_cpu = batch.token_ids_cpu_tensor.numpy()
    for row, (prompt, output) in enumerate(zip(prompts, outputs)):
        batch.token_ids_cpu[row, : len(prompt) + len(output)] = prompt + output
    batch.num_tokens_no_spec = batch.num_prompt_tokens + np.array(list(map(len, outputs)), dtype=np.int32)
    batch.sampled_token_ids_cpu = None
    batch.prev_req_id_to_index = None
    batch.async_copy_ready_event = Mock()
    metadata = SimpleNamespace(
        output_token_ids=outputs,
        prompt_token_ids=torch.tensor(prompts),
        persistent_data={},
        token_history_ids_cpu=None,
        token_history_lens_cpu=None,
    )
    for name, value in dict(
        multiplier=5.0,
        base=1.75,
        allowed_length=1,
        ranges=0,
        max_ngram=12,
        max_occurrences=8,
        early_exit_match_len=8,
    ).items():
        tensor = torch.full((len(prompts),), value, dtype=torch.float32 if isinstance(value, float) else torch.int32)
        setattr(metadata, "dry_" + name, tensor)
        setattr(metadata, "dry_" + name + "_cpu", tensor)
    metadata.dry_sequence_breaker_ids = torch.empty((len(prompts), 0), dtype=torch.long)
    metadata.dry_sequence_breaker_ids_cpu = metadata.dry_sequence_breaker_ids
    batch.sampling_metadata = metadata
    return batch


def assert_native_matches_reference(batch):
    current_platform.import_kernels()
    if not hasattr(torch.ops._C, "dry_scan_penalties"):
        pytest.skip("Native DRY scanner is not built on this platform")
    metadata = batch.sampling_metadata
    logits = torch.zeros((batch.num_reqs, 16))
    # Fail if the native path accidentally falls back to Python.
    with patch("aphrodite.v1.sample.ops._compute_dry_penalties", side_effect=AssertionError("Python fallback")):
        native = SamplingOps().apply_dry(logits.clone(), metadata)
    with patch.object(metadata, "token_history_ids_cpu", None):
        reference = SamplingOps().apply_dry(logits.clone(), metadata)
    torch.testing.assert_close(native, reference, rtol=0, atol=0)
    return native


def test_dry_history_grows_without_batch_changes():
    batch = make_batch([[1, 2]], [[]])
    batch.refresh_dry_token_history()
    metadata = batch.sampling_metadata
    assert metadata.token_history_ids_cpu.shape == (1, 2)

    for token in [1, 2, 1, 2]:
        end = batch.num_tokens_no_spec[0]
        batch.token_ids_cpu[0, end] = token
        batch.num_tokens_no_spec[0] += 1
        batch.req_output_token_ids[0].append(token)
        batch.refresh_dry_token_history()
        assert batch.sampling_metadata is metadata
        assert metadata.token_history_ids_cpu.shape[1] == end + 1
        result = assert_native_matches_reference(batch)
    assert result[0, 1] < 0


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_dry_async_reordering_and_partial_acceptance(accepted):
    batch = make_batch([[1, 2], [3, 4]], [[-1, -1, -1], [-1, -1, -1]])
    batch.prev_req_id_to_index = {"req0": 1, "req1": 0}
    batch.sampled_token_ids_cpu = torch.tensor(
        [[3, 4, 3][:accepted] + [-1] * (3 - accepted), [1, 2, 1][:accepted] + [-1] * (3 - accepted)]
    )
    optimistic_counts = batch.num_tokens_no_spec.copy()
    batch.update_async_output_token_ids()
    batch.refresh_dry_token_history()
    batch.async_copy_ready_event.synchronize.assert_called_once()
    metadata = batch.sampling_metadata
    assert metadata.token_history_lens_cpu.tolist() == [2 + accepted] * 2
    assert metadata.token_history_ids_cpu.tolist() == [
        [1, 2] + [1, 2, 1][:accepted],
        [3, 4] + [3, 4, 3][:accepted],
    ]
    np.testing.assert_array_equal(batch.num_tokens_no_spec, optimistic_counts)
    assert_native_matches_reference(batch)


def test_dry_async_discarded_outputs_limit_copy():
    batch = make_batch([[1, 2]], [[1, -1]])
    batch.prev_req_id_to_index = {"req0": 0}
    batch.sampled_token_ids_cpu = torch.tensor([[2, 9, 9]])
    batch.update_async_output_token_ids()
    batch.refresh_dry_token_history()
    assert batch.sampling_metadata.token_history_ids_cpu.tolist() == [[1, 2, 1, 2]]
    assert batch.token_ids_cpu[0, 4] == -1
    assert assert_native_matches_reference(batch)[0, 1] == -8.75


def test_dry_refresh_skipped_when_disabled():
    batch = make_batch([[1, 2]], [[]])
    batch.dry_reqs.clear()
    with patch.object(batch, "_get_token_history_cpu_views", side_effect=AssertionError("Unnecessary history work")):
        batch.refresh_dry_token_history()


def test_dry_history_survives_compaction_and_slot_reuse():
    batch = InputBatch(
        max_num_reqs=3,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=16,
        block_sizes=[16],
        max_num_blocks_per_req=[4],
    )

    def add(req_id, pair):
        batch.add_request(
            CachedRequestState(
                req_id=req_id,
                prompt_token_ids=pair,
                mm_features=[],
                sampling_params=SamplingParams(dry_multiplier=5, dry_allowed_length=1, dry_sequence_breaker_ids=[]),
                pooling_params=None,
                block_ids=([],),
                generator=None,
                num_computed_tokens=2,
                output_token_ids=pair.copy(),
            )
        )

    add("a", [1, 2])
    add("b", [3, 4])
    add("c", [5, 6])
    batch.refresh_metadata()
    batch.remove_request("b")
    batch.condense()
    batch.refresh_metadata()
    add("d", [7, 8])
    batch.refresh_metadata()
    batch.refresh_dry_token_history()
    assert batch.req_ids == ["a", "c", "d"]
    token_history_ids = batch.sampling_metadata.token_history_ids_cpu
    assert token_history_ids is not None
    assert token_history_ids.tolist() == [[1, 2, 1, 2], [5, 6, 5, 6], [7, 8, 7, 8]]
    assert_native_matches_reference(batch)


@pytest.mark.parametrize("speculative", [False, True])
def test_runner_refreshes_dry_after_resolving_async_tokens(speculative, monkeypatch):
    batch = make_batch([[1, 2]], [[1, -1]])
    batch.prev_req_id_to_index = {"req0": 0}
    batch.sampled_token_ids_cpu = torch.tensor([[2]])
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = batch
    runner.use_async_scheduling = True
    runner._draft_token_req_ids = None
    monkeypatch.setattr(runner, "_get_spec_decode_draft_probs", Mock(return_value=None))

    def sample(*args, **kwargs):
        assert batch.sampling_metadata.token_history_ids_cpu.tolist() == [[1, 2, 1, 2]]
        return "sampled"

    runner.sampler = Mock(side_effect=sample)
    runner.rejection_sampler = Mock(side_effect=sample)
    assert runner._sample(torch.zeros((1, 16)), Mock() if speculative else None) == "sampled"
