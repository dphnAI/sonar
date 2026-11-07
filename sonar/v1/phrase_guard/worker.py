# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def restore_requests(runner, rewinds):
    """Replace only rewound V1 request histories before ordinary cache resume."""
    for req_id, (output_ids, params) in rewinds.items():
        state = runner.requests[req_id]
        runner.input_batch.remove_request(req_id)
        if runner.input_batch.prev_req_id_to_index is not None:
            runner.input_batch.prev_req_id_to_index.pop(req_id, None)
        state.output_token_ids[:] = output_ids
        state.sampling_params = params
        state.persistent_data.clear()
        state.prev_num_draft_len = 0
