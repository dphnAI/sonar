# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aphrodite.platforms import current_platform
from aphrodite.sampling_params import SamplingParams
from aphrodite.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(), reason="requires CUDA row-packing kernels"
)


@pytest.fixture
def state():
    from aphrodite.models.deepseek_v41.sm120.ced import CEDState
    from aphrodite.models.deepseek_v41.sm120.model_state import DeepseekV41ModelState

    # Request bookkeeping and real CED buffers, without loading model weights.
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.rope_state = None
    state.prompt_embeds_state = None
    state.decoder_replay_layers = None
    state._replay_start_np = np.zeros(2, dtype=np.int32)
    state._ced_prompt_logprobs = set()
    state._ced_prefix_start = {}
    state._ced_streaming_prefix_start = None
    state.aphrodite_config = SimpleNamespace(aux_output_config=SimpleNamespace(enable_return_routed_experts=False))
    state.ced_state = CEDState(2, 512, (128, 256, 384, 512), torch.device("cuda"))
    return state


@pytest.mark.parametrize("mode", ["raw_logits", "raw_logprobs"])
def test_fixed_token_prompt_scores_survive_ced_compaction(state, mode):
    from aphrodite.models.deepseek_v41.sm120.ced import gather_rows, scatter_rows

    rows = np.arange(255, dtype=np.int32)
    ids = np.stack((rows % 11, (rows + 3) % 11), axis=1)
    params = SamplingParams(prompt_logprob_token_ids=ids)
    requests = [("scoring", params), ("ordinary", SamplingParams())]
    worker = PromptLogprobsWorker(2, torch.device("cuda"), logprobs_mode=mode)
    for index, (req_id, sampling) in enumerate(requests):
        state.add_request(index, SimpleNamespace(req_id=req_id, sampling_params=sampling, replay_start=0))
        worker.add_request(req_id, index, sampling)
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=[req_id for req_id, _ in requests],
        query_start_loc=torch.tensor([0, 256, 512], dtype=torch.int32, device="cuda"),
        query_start_loc_np=np.array([0, 256, 512], dtype=np.int32),
        seq_lens=torch.tensor([256, 256], dtype=torch.int32, device="cuda"),
        num_scheduled_tokens=np.array([256, 256], dtype=np.int32),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(2, dtype=np.int32),
        prefill_len_np=np.array([256, 256], dtype=np.int32),
    )
    state._stage_ced(batch)
    indices = state.get_ced_indices()
    assert indices is not None
    hidden = (torch.arange(512 * 8, dtype=torch.float32, device="cuda").view(512, 8) % 37 - 18) / 37
    weights = (torch.arange(8 * 11, dtype=torch.float32, device="cuda").view(8, 11) % 17 - 8) / 17
    restored = scatter_rows(gather_rows(hidden, indices), indices, 512)

    def logits_fn(hidden_states):
        return hidden_states @ weights

    scores = worker.compute_prompt_token_id_logprobs(logits_fn, restored, batch, np.array([256, 256]))
    reference = logits_fn(hidden[:255])
    if mode == "raw_logprobs":
        reference = reference.log_softmax(dim=-1)
    expected = reference.gather(-1, torch.as_tensor(ids, device="cuda", dtype=torch.int64))
    torch.testing.assert_close(scores["scoring"], expected)
    # Only the scoring request retains its entire prefix; ordinary generation
    # still retains exactly its trailing decoder window.
    torch.testing.assert_close(
        indices, torch.cat((torch.arange(256, device="cuda"), torch.arange(384, 512, device="cuda")))
    )

    # Reusing a completed scoring request ID must not disable compaction for a
    # subsequent ordinary request.
    state.remove_request("scoring")
    state.add_request(0, SimpleNamespace(req_id="scoring", sampling_params=SamplingParams(), replay_start=0))
    state._stage_ced(batch)
    torch.testing.assert_close(
        state.get_ced_indices(),
        torch.cat((torch.arange(128, 256, device="cuda"), torch.arange(384, 512, device="cuda"))),
    )
