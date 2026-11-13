# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side token and length stops for speculative acceptance.

Mirror the scheduler's EOS, stop-token and length checks. Stop strings and
repetition detection remain scheduler/output-processor responsibilities.
"""

import torch

from sonar.sampling_params import SamplingParams
from sonar.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from sonar.v1.worker.gpu.sample.logit_bias import MAX_NUM_STOP_TOKEN_IDS
from sonar.v1.worker.gpu.states import RequestState


class StopConditions:
    def __init__(self, req_states: RequestState) -> None:
        self.prompt_len_np = req_states.prompt_len.np
        self.max_model_len = req_states.max_model_len
        max_num_reqs = req_states.max_num_reqs
        # -1 when EOS does not stop the request.
        self.eos_token_id = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.max_len = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.num_stop_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.stop_token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_NUM_STOP_TOKEN_IDS),
            dtype=torch.int32,
            device=req_states.device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        prompt_len = int(self.prompt_len_np[req_idx])
        eos_token_id = sampling_params.eos_token_id
        self.eos_token_id.np[req_idx] = -1 if eos_token_id is None else eos_token_id
        max_len = self.max_model_len
        if sampling_params.max_tokens is not None:
            max_len = min(max_len, prompt_len + sampling_params.max_tokens)
        self.max_len.np[req_idx] = max_len
        stop_token_ids = sampling_params.stop_token_ids or []
        if len(stop_token_ids) > MAX_NUM_STOP_TOKEN_IDS:
            raise ValueError(f"Too many stop tokens: {len(stop_token_ids)}. The max size is {MAX_NUM_STOP_TOKEN_IDS}.")
        self.num_stop_token_ids.np[req_idx] = len(stop_token_ids)
        self.stop_token_ids.stage_write(req_idx, 0, stop_token_ids)

    def apply_staged_writes(self) -> None:
        self.eos_token_id.copy_to_uva()
        self.max_len.copy_to_uva()
        self.num_stop_token_ids.copy_to_uva()
        self.stop_token_ids.apply_write()
