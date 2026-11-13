# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from sonar.model_executor.layers.utils import apply_penalties
from sonar.utils.torch_utils import PIN_MEMORY, make_ndarray_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """Applies presence, frequency and repetition penalties to the logits."""
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # In the async scheduling case, rows that won't have penalties applied may contain
    # -1 placeholder token ids. We must replace these with valid token ids so that the
    # scatter done in apply_penalties is valid.
    # NOTE(nick): The penalties implementation is currently quite inefficient and
    # will be reworked anyhow.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors(output_token_ids: list[list[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Convert the different list data structures to tensors."""
    output_tokens_tensor = torch.from_numpy(make_ndarray_with_pad(output_token_ids, pad=vocab_size, dtype=np.int64))
    if PIN_MEMORY:
        output_tokens_tensor = output_tokens_tensor.pin_memory()
    return output_tokens_tensor.to(device, non_blocking=True)
