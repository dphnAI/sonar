# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from sonar.triton_utils import tl, triton
from sonar.v1.phrase_guard.matcher import MAX_RETRIES
from sonar.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RetryMask:
    def __init__(self, max_num_reqs, device):
        self.positions = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.counts = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.tokens = StagedWriteTensor((max_num_reqs, MAX_RETRIES), dtype=torch.int32, device=device)

    def add_request(self, index, prompt_len, retry):
        self.counts.np[index] = 0
        if retry is not None:
            position, tokens = retry
            self.positions.np[index] = prompt_len + position - 1
            self.counts.np[index] = len(tokens)
            self.tokens.stage_write(index, 0, tokens)

    def apply_staged_writes(self):
        self.positions.copy_to_uva()
        self.counts.copy_to_uva()
        self.tokens.apply_write()

    def apply(self, logits, expanded_idx_mapping, idx_mapping_np, positions):
        if not np.any(self.counts.np[idx_mapping_np]):
            return
        _mask[(logits.shape[0],)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            positions,
            self.positions.gpu,
            self.counts.gpu,
            self.tokens.gpu,
            WIDTH=MAX_RETRIES,
            VOCAB_SIZE=logits.shape[1],
            BLOCK_SIZE=min(4096, triton.next_power_of_2(logits.shape[1])),
        )


class CPURetryMask:
    """RetryMask for the standalone CPU model runner, which runs without Triton.

    Same semantics as `_mask`: at the retry position, block the rejected tokens
    and, if that leaves no finite logit, reopen the first one.
    """

    def __init__(self, max_num_reqs):
        self.positions = np.zeros(max_num_reqs, dtype=np.int64)
        self.counts = np.zeros(max_num_reqs, dtype=np.int32)
        self.tokens = np.zeros((max_num_reqs, MAX_RETRIES), dtype=np.int64)

    def add_request(self, index, prompt_len, retry):
        self.counts[index] = 0
        if retry is not None:
            position, tokens = retry
            self.positions[index] = prompt_len + position - 1
            self.counts[index] = len(tokens)
            self.tokens[index, : len(tokens)] = tokens

    def apply(self, logits, expanded_idx_mapping, idx_mapping_np, positions):
        if not np.any(self.counts[idx_mapping_np]):
            return
        for row, (request, position) in enumerate(zip(expanded_idx_mapping.tolist(), positions.tolist())):
            count = int(self.counts[request])
            if count == 0 or position != self.positions[request]:
                continue
            tokens = torch.from_numpy(self.tokens[request, :count]).to(logits.device)
            logits[row, tokens] = float("-inf")
            if logits[row].max() == float("-inf"):
                # Match V1's blocked-token sentinel.
                logits[row, int(self.tokens[request, 0])] = 0.0


@triton.jit
def _mask(
    logits,
    stride,
    mapping,
    positions,
    checkpoints,
    counts,
    tokens,
    WIDTH: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(mapping + row)
    count = tl.load(counts + request)
    if count > 0 and tl.load(positions + row) == tl.load(checkpoints + request):
        offsets = tl.arange(0, WIDTH)
        valid = offsets < count
        ids = tl.load(tokens + request * WIDTH + offsets, valid, other=0)
        tl.store(logits + row * stride + ids, float("-inf"), valid)
        tl.debug_barrier()
        vocab = tl.arange(0, BLOCK_SIZE)
        maximum = float("-inf")
        start = 0
        while start < VOCAB_SIZE and maximum == float("-inf"):
            columns = start + vocab
            values = tl.load(logits + row * stride + columns, columns < VOCAB_SIZE, other=float("-inf"))
            maximum = tl.max(values, 0)
            start += BLOCK_SIZE
        if maximum == float("-inf"):
            # Match V1's blocked-token sentinel without a GPU-to-CPU sync.
            first = tl.load(tokens + request * WIDTH)
            tl.store(logits + row * stride + first, 0.0)
