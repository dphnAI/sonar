# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.triton_utils import tl, tldevice, triton
from aphrodite.v1.worker.gpu.sample.gumbel import (
    gumbel_block_argmax,
    spec_step_seed,
    tl_rand32,
)
from aphrodite.v1.worker.gpu.sample.vocab_parallel import VocabShard
from aphrodite.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax


@triton.jit
def _compute_max_and_sumexp(logits):
    max = tl.max(logits, axis=0)
    sumexp = tl.where(
        max > float("-inf"),
        tl.sum(tl.exp(logits - max)),
        0.0,
    )
    return max, sumexp


@triton.jit
def _compute_global_logsumexp(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    global_lse = global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))
    return global_lse


@triton.jit
def _compute_global_residual_mass(
    local_residual_mass_ptr,
    local_residual_mass_stride,
    prefix_joint_ratio,
    target_logits_ptr,
    target_logits_stride,
    token_logits_ptr,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_token,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    if HAS_DRAFT_LOGITS:
        blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
        mask = blocks < vocab_num_blocks
        partials = tl.load(
            local_residual_mass_ptr + logit_idx * local_residual_mass_stride + blocks,
            mask=mask,
            other=0.0,
        )
        return tl.sum(partials, axis=0)
    else:
        # One-hot draft. M_s is a point mass at this draft token
        # so the residual mass reduces to the closed form:
        #   p * (1 - M_b(draft_token)).
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        if token_logits_ptr is not None:
            target_logit = tl.load(token_logits_ptr + 2 * logit_idx)
        else:
            target_logit = tl.load(
                target_logits_ptr + logit_idx * target_logits_stride + draft_token,
            ).to(tl.float32)
        m_b = tl.exp(target_logit - target_lse)
        return prefix_joint_ratio * (1.0 - m_b)


@triton.jit
def _compute_global_target_argmax(
    target_local_max_ptr,
    target_local_max_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    local_max = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    # See _insert_resampled_kernel: NaN breaks tl.argmax index bounds.
    local_max = tl.where(local_max != local_max, float("-inf"), local_max)
    max_block_idx = tl.argmax(local_max, axis=0)
    return tl.load(target_local_argmax_ptr + logit_idx * target_local_argmax_stride + max_block_idx).to(tl.int64)


@triton.jit
def _compute_global_logprobs_and_logsumexp(
    token,
    mask,
    logit_idx,
    req_state_idx,
    draft_step,
    temp,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, 2] or None. Vocab-parallel only: the target and draft
    # logits of the row's next draft token, read instead of `token`'s logits.
    token_logits_ptr,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    if token_logits_ptr is not None:
        target_logit = tl.load(token_logits_ptr + 2 * logit_idx)
    else:
        target_logit = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + token,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
    target_lse = _compute_global_logsumexp(
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    target_log_prob = target_logit - target_lse
    if HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.
        if token_logits_ptr is not None:
            draft_logit = tl.load(token_logits_ptr + 2 * logit_idx + 1) / temp
        else:
            draft_logit = (
                tl.load(
                    draft_logits_ptr
                    + req_state_idx * draft_logits_stride_0
                    + draft_step * draft_logits_stride_1
                    + token,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
        draft_lse = _compute_global_logsumexp(
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        draft_log_prob = draft_logit - draft_lse
    else:
        # One-hot draft: q(token) = 1, log_q = 0.
        draft_log_prob = 0.0
        draft_lse = 0.0
    return target_log_prob, draft_log_prob, target_lse, draft_lse


@triton.jit
def _is_bonus_logit(expanded_local_pos_ptr, logit_idx, num_logits):
    next_local_pos = tl.load(
        expanded_local_pos_ptr + logit_idx + 1,
        mask=logit_idx + 1 < num_logits,
        other=0,
    )
    return next_local_pos == 0


@triton.jit
def _compute_local_logits_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits, 2] or None. Vocab-parallel only: receives this shard's
    # target and draft logits of each row's next draft token.
    token_logits_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_logits,
    # Global token id of local column 0.
    token_offset,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)

    is_bonus = _is_bonus_logit(expanded_local_pos_ptr, logit_idx, num_logits)
    if token_logits_ptr is None and is_bonus:
        # Max/argmax and summed exponentials are not needed.
        return

    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if token_logits_ptr is not None:
        if block_idx == 0:
            # Kernels that read a single token's logit cannot see other
            # shards, so the owning shard publishes it; the rest store -inf.
            next_token = tl.load(
                draft_sampled_ptr + logit_idx + 1,
                mask=not is_bonus,
                other=-1,
            ).to(tl.int64)
            local_token = next_token - token_offset
            is_owned = (local_token >= 0) & (local_token < vocab_size)
            target_token_logit = tl.load(
                target_logits_ptr + logit_idx * target_logits_stride + local_token,
                mask=is_owned,
                other=float("-inf"),
            ).to(tl.float32)
            tl.store(token_logits_ptr + 2 * logit_idx, target_token_logit)
            if HAS_DRAFT_LOGITS:
                draft_token_logit = tl.load(
                    draft_logits_ptr
                    + req_state_idx * draft_logits_stride_0
                    + draft_step_idx * draft_logits_stride_1
                    + local_token,
                    mask=is_owned,
                    other=float("-inf"),
                ).to(tl.float32)
                tl.store(token_logits_ptr + 2 * logit_idx + 1, draft_token_logit)
        if is_bonus and temp != 0.0:
            # Keep the greedy bonus argmax so all-greedy batches can take it
            # from these stats instead of exchanging resample partials.
            return

    if temp == 0.0:
        # Greedy sampling. Only the target max/argmax are needed.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx + token_offset
        tl.store(
            target_local_argmax_ptr + logit_idx * target_local_argmax_stride + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        # Get local target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr + logit_idx * target_local_sumexp_stride + block_idx,
            target_sumexp,
        )
        if HAS_DRAFT_LOGITS:
            # Get local draft max and summed exponentials. draft_logits is
            # stored pre-temperature, so apply scale first.
            draft_logits = (
                tl.load(
                    draft_logits_ptr
                    + req_state_idx * draft_logits_stride_0
                    + draft_step_idx * draft_logits_stride_1
                    + block_offsets,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
            draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr + logit_idx * draft_local_sumexp_stride + block_idx,
                draft_sumexp,
            )


@triton.jit
def _compute_cumulative_log_p_kernel(
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, 2] or None (see _compute_global_logprobs_and_logsumexp)
    token_logits_ptr,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    log_p = tl.zeros((), tl.float32)
    for step in range(num_draft_tokens):
        logit_idx = start_idx + step
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 placeholder tokens can never be accepted. Skip their reductions
        # and carry the last valid cumulative value.
        if draft_token >= 0:
            target_logprob, draft_logprob, _, _ = _compute_global_logprobs_and_logsumexp(
                draft_token,
                True,  # mask
                logit_idx,
                req_state_idx,
                step,
                temp,
                target_logits_ptr,
                target_logits_stride,
                token_logits_ptr,
                target_local_max_ptr,
                target_local_max_stride,
                target_local_sumexp_ptr,
                target_local_sumexp_stride,
                draft_logits_ptr,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
                HAS_DRAFT_LOGITS,
            )
            log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)
        tl.store(cumulative_log_p_ptr + logit_idx, log_p)


@triton.jit
def _compute_local_residual_mass_kernel(
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_logits,
    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    # Adaptive verification can end before num_speculative_steps.
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx == 0 or _is_bonus_logit(expanded_local_pos_ptr, logit_idx, num_logits):
        # The acceptance threshold, h, looks one position ahead and sums
        # over: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0). Tokens at the
        # first and last (bonus) positions aren't needed for this computation.
        return

    if tl.load(draft_sampled_ptr + logit_idx + 1) < 0:
        # -1 placeholder token. The rejection kernel treats the preceding token
        # as the end of the block, so this position's residual mass is unused.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size
    target_log_probs, draft_log_probs, _, _ = _compute_global_logprobs_and_logsumexp(
        block_offsets,
        mask,
        logit_idx,
        req_state_idx,
        draft_step_idx,
        temp,
        target_logits_ptr,
        target_logits_stride,
        None,  # token_logits_ptr: this sweeps the local block's logits
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        draft_logits_ptr,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max_ptr,
        draft_local_max_stride,
        draft_local_sumexp_ptr,
        draft_local_sumexp_stride,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
        True,  # HAS_DRAFT_LOGITS
    )

    # Compute the residual mass: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0)
    p = tl.exp(tl.load(cumulative_log_p_ptr + logit_idx - 1).to(tl.float32))
    m_b = tl.exp(target_log_probs)
    m_s = tl.exp(draft_log_probs)
    partial = tl.sum(tl.maximum(p * m_b - m_s, 0.0), axis=0)
    tl.store(
        local_residual_mass_ptr + logit_idx * local_residual_mass_stride + block_idx,
        partial,
    )


@triton.jit
def _rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, 2] or None (see _compute_global_logprobs_and_logsumexp)
    token_logits_ptr,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 is used for placeholder draft token ids that should be rejected.
        is_valid_draft = draft_sampled >= 0
        # Avoid possible OOB ptr access.
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            # A -1 placeholder ends verification. Greedy is excluded because it
            # stores the target argmax upon first rejection, so it rejects the
            # placeholder via `accepted` instead.
            verifying &= is_valid_draft
        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            step_seed = spec_step_seed(seed, i) if USE_BLOCK_VERIFICATION else seed
            u = tl_rand32(step_seed, pos, includes_zero=False)
            if is_greedy:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    accepted = target_argmax == draft_sampled
                accepted &= is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            elif USE_BLOCK_VERIFICATION:
                # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444
                prefix_joint_ratio = tl.exp(tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32))
                next_draft_token = tl.load(
                    draft_sampled_ptr + logit_idx + 2,
                    mask=i < num_draft_tokens - 1,
                    other=-1,
                ).to(tl.int64)
                if next_draft_token >= 0:
                    residual_mass = _compute_global_residual_mass(
                        local_residual_mass_ptr,
                        local_residual_mass_stride,
                        prefix_joint_ratio,
                        target_logits_ptr,
                        target_logits_stride,
                        token_logits_ptr,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        next_draft_token,
                        logit_idx + 1,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted_length = tl.where(u <= h, i + 1, accepted_length)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            else:
                # Speculative decoding (Leviathan et al., 2023): https://arxiv.org/abs/2211.17192
                target_logprob, draft_logprob, target_lse, draft_lse = _compute_global_logprobs_and_logsumexp(
                    draft_sampled,
                    True,  # mask
                    logit_idx,
                    req_state_idx,
                    i,
                    temp,
                    target_logits_ptr,
                    target_logits_stride,
                    token_logits_ptr,
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    draft_logits_ptr,
                    draft_logits_stride_0,
                    draft_logits_stride_1,
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                    HAS_DRAFT_LOGITS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_logprob > tl.log(u) + draft_logprob
                verifying = accepted
                accepted_length += accepted
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length < num_draft_tokens:
        # Compute the target and draft log exponential sums for the
        # rejected token.
        rejected_idx = start_idx + accepted_length
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        if HAS_DRAFT_LOGITS:
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                rejected_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _seeded_resample_argmax(
    residual_logits,
    block,
    mask,
    resample_token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    vocab_size,
    USE_FP64: tl.constexpr,
):
    return gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,  # logits_cache_ptr
        0,  # logits_cache_stride_0
        0,  # logits_cache_stride_1
        None,  # logits_cache_col_ptr
        None,  # logits_cache_source_ptr
        0,  # logits_cache_source_stride
        vocab_size,
        IS_DRAFTING=False,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )


@triton.jit(do_not_specialize=["watermark_key_0", "watermark_key_1"])
def _resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, CONTEXT_WIDTH]
    contexts_ptr,
    contexts_stride,
    # [max_num_reqs], uint8 view of a bool tensor
    watermarking_ptr,
    watermarking_skip_mask_ptr,
    watermark_key_0,
    watermark_key_1,
    vocab_size,
    # Global token id of local column 0.
    token_offset,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
    WATERMARK: tl.constexpr,
    DEDUPLICATE_CONTEXTS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. No resampling needed because
        # the target argmax is already in the sampled tensor.
        return

    rejected_draft_token = tl.load(
        draft_sampled_ptr + resample_token_idx + 1,
        mask=not is_bonus,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    # Global token ids. They key the Gumbel noise, so a vocab shard draws the
    # same per-token noise as the full vocabulary.
    tokens = block + token_offset
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # Compute the residual logits to resample the rejected token from.
    is_watermarked = False
    if WATERMARK:
        is_watermarked = (
            tl.load(
                watermarking_ptr + req_state_idx,
                mask=req_state_idx >= 0,
                other=0,
            )
            != 0
        )

    skip_watermarking = False
    if DEDUPLICATE_CONTEXTS:
        skip_watermarking = (
            (tl.load(watermarking_skip_mask_ptr + resample_token_idx) != 0) & is_watermarked & (temp != 0.0)
        )

    if is_bonus or not is_valid_rejected_draft:
        # Bonus token (no rejections) or -1 placeholder token. In either case,
        # directly use the target logits.
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.
        draft_logits = (
            tl.load(
                draft_logits_ptr + req_state_idx * draft_logits_stride_0 + resample_idx * draft_logits_stride_1 + block,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        if USE_BLOCK_VERIFICATION:
            # Block residual is:
            #   max(p_tau * M_b(x) - M_s(x), 0) / Z.
            # Scale the target logprobs by log(p_tau). p_0 = 1, so skip
            # shifting when nothing was accepted (tau == 0).
            log_p_tau = 0.0
            if resample_idx > 0:
                log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(tl.float32)
            target_log_probs += log_p_tau
        draft_log_probs = draft_logits - draft_lse
        # Compute the residual:
        #   r(x) = max(p(x) - q(x), 0)
        # Gumbel sampling needs logits, so we compute it in log space:
        #   log(r(x)) = log(max(exp(log_p(x)) - exp(log_q(x)), 0))
        # The more numerically stable form is:
        #   log(max(exp(a) - exp(b), 0)) = a + log(max(1 - exp(b - a), 0))
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        # One-hot draft. The residual is just the target distribution with
        # the rejected draft token probability zeroed out.
        # NOTE: During block verification, the residual becomes:
        #   0                   if x == rejected_draft_token
        #   p_tau * M_b(x) / Z  otherwise
        # Therefore p_tau is a constant that cancels under normalization,
        # and does not need to be applied.
        residual_logits = tl.where(
            tokens != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    # Resample the rejected/bonus token.
    if WATERMARK:
        # Padded and greedy rows retain the stock draw.
        if is_watermarked & (temp != 0.0) & ~skip_watermarking:
            watermark_value, idx = philox_gumbel_block_argmax(
                residual_logits,
                mask,
                block_idx,
                contexts_ptr + resample_token_idx * contexts_stride,
                watermark_key_0.to(tl.uint32),
                watermark_key_1.to(tl.uint32),
                CONTEXT_WIDTH,
                BLOCK_SIZE,
            )
            # Detector compatibility fixes the keyed draw at fp32.
            value = watermark_value.to(tl.float64) if USE_FP64 else watermark_value
        else:
            value, idx = _seeded_resample_argmax(
                residual_logits,
                tokens,
                mask,
                resample_token_idx,
                expanded_idx_mapping_ptr,
                temp_ptr,
                seed_ptr,
                pos_ptr,
                vocab_size,
                USE_FP64=USE_FP64,
            )
    else:
        value, idx = _seeded_resample_argmax(
            residual_logits,
            tokens,
            mask,
            resample_token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seed_ptr,
            pos_ptr,
            vocab_size,
            USE_FP64=USE_FP64,
        )
    token_id = block_idx * BLOCK_SIZE + idx + token_offset
    tl.store(
        resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        return

    # Insert the resampled token.
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # NaN max values (from NaN target logits) make tl.argmax return an
    # out-of-range block index (into the padded region), causing an OOB read
    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.
    resampled_local_max = tl.where(
        resampled_local_max != resampled_local_max,
        float("-inf"),
        resampled_local_max,
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + resampled_max_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        resampled,
    )


# Per-block logits stats reduce the vocabulary in blocks of this many tokens.
_VOCAB_BLOCK_SIZE = 8192
# Resampling draws the per-block Gumbel argmax over blocks of this many tokens.
_RESAMPLE_BLOCK_SIZE = 1024


def _draft_logits_strides(draft_logits: torch.Tensor | None) -> tuple[int, int]:
    if draft_logits is None:
        return 0, 0
    return draft_logits.stride(0), draft_logits.stride(1)


def _launch_local_logits_stats(
    # [num_logits, num_blocks]
    target_local_argmax: torch.Tensor,
    target_local_max: torch.Tensor,
    target_local_sumexp: torch.Tensor,
    draft_local_max: torch.Tensor,
    draft_local_sumexp: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_sampled: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    vocab_size: int,
    token_logits: torch.Tensor | None = None,
    token_offset: int = 0,
) -> None:
    num_logits, num_blocks = target_local_max.shape
    draft_logits_stride_0, draft_logits_stride_1 = _draft_logits_strides(draft_logits)
    _compute_local_logits_stats_kernel[(num_logits, num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        expanded_idx_mapping,
        expanded_local_pos,
        draft_sampled,
        token_logits,
        temperature,
        vocab_size,
        num_logits,
        token_offset,
        BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=draft_logits is not None,
    )


def _launch_cumulative_log_p(
    # [num_logits]
    cumulative_log_p: torch.Tensor,
    target_logits: torch.Tensor,
    # [num_logits, vocab_num_blocks]
    target_local_max: torch.Tensor,
    target_local_sumexp: torch.Tensor,
    draft_sampled: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_local_max: torch.Tensor,
    draft_local_sumexp: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    token_logits: torch.Tensor | None = None,
) -> None:
    vocab_num_blocks = target_local_max.shape[1]
    draft_logits_stride_0, draft_logits_stride_1 = _draft_logits_strides(draft_logits)
    _compute_cumulative_log_p_kernel[(cu_num_logits.shape[0] - 1,)](
        cumulative_log_p,
        target_logits,
        target_logits.stride(0),
        token_logits,
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(vocab_num_blocks),
        HAS_DRAFT_LOGITS=draft_logits is not None,
        num_warps=1,
    )


def _launch_local_residual_mass(
    # [num_logits, num_blocks], one partial per block of target_logits
    local_residual_mass: torch.Tensor,
    cumulative_log_p: torch.Tensor,
    target_logits: torch.Tensor,
    # [num_logits, vocab_num_blocks], reduced to the logsumexp
    target_local_max: torch.Tensor,
    target_local_sumexp: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_local_max: torch.Tensor,
    draft_local_sumexp: torch.Tensor,
    draft_sampled: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    vocab_size: int,
) -> None:
    num_logits, num_blocks = local_residual_mass.shape
    vocab_num_blocks = target_local_max.shape[1]
    _compute_local_residual_mass_kernel[(num_logits, num_blocks)](
        local_residual_mass,
        local_residual_mass.stride(0),
        cumulative_log_p,
        target_logits,
        target_logits.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        draft_sampled,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_logits,
        vocab_num_blocks,
        BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(vocab_num_blocks),
    )


def _launch_rejection(
    # [num_reqs, num_speculative_steps + 1]
    sampled: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    target_rejected_logsumexp: torch.Tensor,
    draft_rejected_logsumexp: torch.Tensor,
    target_logits: torch.Tensor,
    # [num_logits, vocab_num_blocks]
    target_local_argmax: torch.Tensor,
    target_local_max: torch.Tensor,
    target_local_sumexp: torch.Tensor,
    draft_sampled: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_local_max: torch.Tensor,
    draft_local_sumexp: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    cumulative_log_p: torch.Tensor | None,
    # [num_logits, vocab_num_blocks]
    local_residual_mass: torch.Tensor | None,
    use_block_verification: bool,
    synthetic_conditional_rates: torch.Tensor | None = None,
    token_logits: torch.Tensor | None = None,
) -> None:
    vocab_num_blocks = target_local_max.shape[1]
    draft_logits_stride_0, draft_logits_stride_1 = _draft_logits_strides(draft_logits)
    _rejection_kernel[(sampled.shape[0],)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        token_logits,
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        cumulative_log_p,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(vocab_num_blocks),
        HAS_DRAFT_LOGITS=draft_logits is not None,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )


def _launch_resample(
    # [num_reqs, num_blocks]
    resampled_local_argmax: torch.Tensor,
    resampled_local_max: torch.Tensor,
    target_logits: torch.Tensor,
    target_rejected_logsumexp: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_rejected_logsumexp: torch.Tensor,
    num_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    draft_sampled: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    cumulative_log_p: torch.Tensor | None,
    vocab_size: int,
    use_fp64: bool,
    use_block_verification: bool,
    token_offset: int = 0,
    contexts: torch.Tensor | None = None,
    watermarking_bytes: torch.Tensor | None = None,
    watermarking_skip_mask_bytes: torch.Tensor | None = None,
    watermark_key_0: int = 0,
    watermark_key_1: int = 0,
) -> None:
    num_reqs, num_blocks = resampled_local_max.shape
    draft_logits_stride_0, draft_logits_stride_1 = _draft_logits_strides(draft_logits)
    _resample_kernel[(num_reqs, num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        contexts,
        contexts.stride(0) if contexts is not None else 0,
        watermarking_bytes,
        watermarking_skip_mask_bytes,
        watermark_key_0,
        watermark_key_1,
        vocab_size,
        token_offset,
        BLOCK_SIZE=_RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=draft_logits is not None,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
        CONTEXT_WIDTH=contexts.shape[-1] if contexts is not None else 1,
        WATERMARK=contexts is not None,
        DEDUPLICATE_CONTEXTS=watermarking_skip_mask_bytes is not None,
    )


def _launch_insert_resampled(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    # [num_reqs, resample_num_blocks]
    resampled_local_argmax: torch.Tensor,
    resampled_local_max: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_reqs, resample_num_blocks = resampled_local_max.shape
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=triton.next_power_of_2(resample_num_blocks),
    )


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
    contexts: torch.Tensor | None = None,
    watermarking: torch.Tensor | None = None,
    watermarking_skip_mask: torch.Tensor | None = None,
    watermark_key: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    assert draft_logits is None or (draft_logits.ndim == 3 and draft_logits.stride(-1) == 1)
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    watermark = contexts is not None
    assert watermark == (watermarking is not None) == (watermark_key is not None), (
        "contexts, watermarking and watermark_key must be set together."
    )
    assert watermarking_skip_mask is None or watermark, "watermarking_skip_mask requires watermarking."
    watermarking_bytes: torch.Tensor | None = None
    watermarking_skip_mask_bytes: torch.Tensor | None = None
    watermark_key_0 = 0
    watermark_key_1 = 0
    if contexts is not None:
        assert watermarking is not None and watermark_key is not None
        assert contexts.ndim == 2 and contexts.shape[0] == num_logits
        # Context words are hashed as uint32, so int32 (the request-state token
        # dtype) and int64 both land on the same PRF stream, -1 included.
        assert contexts.dtype in (torch.int32, torch.int64)
        if contexts.stride(-1) != 1:
            contexts = contexts.contiguous()
        assert watermarking.ndim == 1 and watermarking.dtype == torch.bool
        watermarking_bytes = watermarking.view(torch.uint8)
        if watermarking_skip_mask is not None:
            assert watermarking_skip_mask.shape == (num_logits,)
            assert watermarking_skip_mask.dtype == torch.bool
            if not watermarking_skip_mask.is_contiguous():
                watermarking_skip_mask = watermarking_skip_mask.contiguous()
            watermarking_skip_mask_bytes = watermarking_skip_mask.view(torch.uint8)
        watermark_key_0 = watermark_key & 0xFFFFFFFF
        watermark_key_1 = watermark_key >> 32
    if draft_logits is not None:
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    vocab_num_blocks = triton.cdiv(vocab_size, _VOCAB_BLOCK_SIZE)
    target_local_argmax = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.int64)
    target_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    target_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    _launch_local_logits_stats(
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        draft_local_max,
        draft_local_sumexp,
        target_logits,
        draft_logits,
        draft_sampled,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
    )

    # Precompute the running joint ratio and residual mass for block
    # verification.
    cumulative_log_p = None
    local_residual_mass = None
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

        # Compute the log of the running joint ratio, p_i.
        # cumulative_log_p[start + i] = log(p_{i+1}), the cumulative ratio after
        # the (i+1)-th draft token.
        cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
        _launch_cumulative_log_p(
            cumulative_log_p,
            target_logits,
            target_local_max,
            target_local_sumexp,
            draft_sampled,
            draft_logits,
            draft_local_max,
            draft_local_sumexp,
            cu_num_logits,
            idx_mapping,
            temperature,
        )

        # Compute the per-vocab-block partials of the residual mass, later reduced
        # to the total by _compute_global_residual_mass. Only launched for full
        # draft logits distributions. One-hot drafts used a closed-form residual
        # mass instead.
        if draft_logits is not None:
            local_residual_mass = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
            _launch_local_residual_mass(
                local_residual_mass,
                cumulative_log_p,
                target_logits,
                target_local_max,
                target_local_sumexp,
                draft_logits,
                draft_local_max,
                draft_local_sumexp,
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
            )

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64)
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    _launch_rejection(
        sampled,
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        draft_sampled,
        draft_logits,
        draft_local_max,
        draft_local_sumexp,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        local_residual_mass,
        use_block_verification,
        synthetic_conditional_rates=synthetic_conditional_rates,
    )

    # Resample the rejected/bonus tokens.
    resample_num_blocks = triton.cdiv(vocab_size, _RESAMPLE_BLOCK_SIZE)
    resampled_local_argmax = target_logits.new_empty(num_reqs, resample_num_blocks, dtype=torch.int64)
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _launch_resample(
        resampled_local_argmax,
        resampled_local_max,
        target_logits,
        target_rejected_logsumexp,
        draft_logits,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        vocab_size,
        use_fp64,
        use_block_verification,
        contexts=contexts,
        watermarking_bytes=watermarking_bytes,
        watermarking_skip_mask_bytes=watermarking_skip_mask_bytes,
        watermark_key_0=watermark_key_0,
        watermark_key_1=watermark_key_1,
    )

    # Insert the resampled tokens into the output sampled.
    _launch_insert_resampled(
        sampled,
        num_sampled,
        resampled_local_argmax,
        resampled_local_max,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
    )
    return sampled, num_sampled


# Per-block logits stats exchanged by vocab-parallel rejection sampling, in
# packet order: target argmax (int32 bits), target max, target sumexp, draft
# max and draft sumexp.
_NUM_LOGITS_STATS = 5
# Integer fields travel inside float packets as same-width bit patterns.
_INT_VIEW_DTYPES = {torch.float32: torch.int32, torch.float64: torch.int64}


def combine_vocab_blocks(gathered: torch.Tensor) -> torch.Tensor:
    """Lay all-gathered per-rank block partials out in ascending vocab order.

    ``[world_size, fields, rows, blocks] -> [fields, rows, world_size * blocks]``.
    Rank ``r``'s block ``b`` becomes global block ``r * blocks + b``, so a
    first-max reduction over the blocks still resolves ties to the lowest
    token id. The copy runs on an integer view, so integer fields packed into
    the float tensor survive bit for bit.
    """
    world_size, fields, rows, blocks = gathered.shape
    bits = gathered.view(_INT_VIEW_DTYPES[gathered.dtype])
    return bits.permute(1, 2, 0, 3).reshape(fields, rows, world_size * blocks).view(gathered.dtype)


def combine_logits_stats(
    # [world_size, packet], from VocabParallelRejection.stats_packet
    gathered: torch.Tensor,
    num_logits: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine the all-gathered logits stats packets of every vocab shard.

    Returns the per-block stats ``[5, num_logits, world_size * num_blocks]``
    (see ``_NUM_LOGITS_STATS``) and the ``[num_logits, 2]`` target and draft
    logits of each row's next draft token.
    """
    world_size = gathered.shape[0]
    num_stats = _NUM_LOGITS_STATS * num_logits * num_blocks
    stats = combine_vocab_blocks(gathered[:, :num_stats].view(world_size, _NUM_LOGITS_STATS, num_logits, num_blocks))
    # Only the shard that owns a token sends its logit; the others send -inf.
    token_logits = gathered[:, num_stats:].view(world_size, num_logits, 2).amax(dim=0)
    return stats, token_logits


class VocabParallelRejection:
    """Rejection sampling over one rank's vocab shard.

    The phases run in this order. Each ``*_packet`` result is all-gathered
    across the shard's ranks (rank order == ascending vocab order) and the
    stacked ``[world_size, *packet.shape]`` tensor is passed to the next step:

    1. ``stats_packet`` -> ``apply_stats``
    2. ``residual_mass_packet`` -> ``apply_residual_mass``, if ``needs_residual_mass``
    3. ``reject``
    4. ``resample_packet`` -> ``finish``, or ``finish(None)`` if not ``needs_resample``

    Everything computed from gathered data runs identically on every rank, so
    all ranks return the same tokens. ``rejection_sample_vocab_parallel`` wires
    the phases with ``VocabShard.all_gather``.
    """

    def __init__(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
        shard: VocabShard,
        *,
        use_fp64: bool = False,
        use_block_verification: bool = False,
        all_greedy: bool = False,
    ):
        assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
        assert target_logits.shape[1] == shard.size
        if all_greedy:
            # Greedy rows read neither the draft distribution nor any block
            # verification state.
            draft_logits = None
            use_block_verification = False
        assert draft_logits is None or (
            draft_logits.ndim == 3 and draft_logits.stride(-1) == 1 and draft_logits.shape[-1] == shard.size
        )
        self.target_logits = target_logits
        self.draft_logits = draft_logits
        self.draft_sampled = draft_sampled
        self.cu_num_logits = cu_num_logits
        self.pos = pos
        self.idx_mapping = idx_mapping
        self.expanded_idx_mapping = expanded_idx_mapping
        self.expanded_local_pos = expanded_local_pos
        self.temperature = temperature
        self.seed = seed
        self.num_speculative_steps = num_speculative_steps
        self.shard = shard
        self.use_fp64 = use_fp64
        self.use_block_verification = use_block_verification
        self.num_reqs = cu_num_logits.shape[0] - 1
        self.num_logits = target_logits.shape[0]
        self.num_blocks = triton.cdiv(shard.size, _VOCAB_BLOCK_SIZE)
        self.needs_residual_mass = use_block_verification and draft_logits is not None
        # All-greedy batches take their bonus tokens from the stats exchange.
        self.needs_resample = not all_greedy
        self.cumulative_log_p: torch.Tensor | None = None
        self.local_residual_mass: torch.Tensor | None = None

    def stats_packet(self) -> torch.Tensor:
        """This shard's per-block logits stats and next-draft-token logits."""
        num_stats = _NUM_LOGITS_STATS * self.num_logits * self.num_blocks
        packet = self.target_logits.new_empty(num_stats + 2 * self.num_logits, dtype=torch.float32)
        stats = packet[:num_stats].view(_NUM_LOGITS_STATS, self.num_logits, self.num_blocks)
        _launch_local_logits_stats(
            stats[0].view(torch.int32),
            stats[1],
            stats[2],
            stats[3],
            stats[4],
            self.target_logits,
            self.draft_logits,
            self.draft_sampled,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.temperature,
            self.shard.size,
            token_logits=packet[num_stats:].view(self.num_logits, 2),
            token_offset=self.shard.start,
        )
        return packet

    def apply_stats(self, gathered: torch.Tensor) -> None:
        self.stats, self.token_logits = combine_logits_stats(gathered, self.num_logits, self.num_blocks)
        if self.use_block_verification:
            self.cumulative_log_p = self.target_logits.new_empty(self.num_logits, dtype=torch.float32)
            _launch_cumulative_log_p(
                self.cumulative_log_p,
                self.target_logits,
                self.stats[1],
                self.stats[2],
                self.draft_sampled,
                self.draft_logits,
                self.stats[3],
                self.stats[4],
                self.cu_num_logits,
                self.idx_mapping,
                self.temperature,
                token_logits=self.token_logits,
            )

    def residual_mass_packet(self) -> torch.Tensor:
        """This shard's per-block residual mass partials, ``[1, num_logits, num_blocks]``."""
        assert self.needs_residual_mass and self.draft_logits is not None and self.cumulative_log_p is not None
        packet = self.target_logits.new_empty(1, self.num_logits, self.num_blocks, dtype=torch.float32)
        _launch_local_residual_mass(
            packet[0],
            self.cumulative_log_p,
            self.target_logits,
            self.stats[1],
            self.stats[2],
            self.draft_logits,
            self.stats[3],
            self.stats[4],
            self.draft_sampled,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.temperature,
            self.shard.size,
        )
        return packet

    def apply_residual_mass(self, gathered: torch.Tensor) -> None:
        self.local_residual_mass = combine_vocab_blocks(gathered)[0]

    def reject(self) -> None:
        self.sampled = self.draft_sampled.new_empty(self.num_reqs, self.num_speculative_steps + 1, dtype=torch.int64)
        self.num_sampled = self.sampled.new_empty(self.num_reqs, dtype=torch.int32)
        self.target_rejected_logsumexp = self.target_logits.new_empty(self.num_reqs, dtype=torch.float32)
        self.draft_rejected_logsumexp = self.target_logits.new_empty(self.num_reqs, dtype=torch.float32)
        _launch_rejection(
            self.sampled,
            self.num_sampled,
            self.target_rejected_logsumexp,
            self.draft_rejected_logsumexp,
            self.target_logits,
            self.stats[0].view(torch.int32),
            self.stats[1],
            self.stats[2],
            self.draft_sampled,
            self.draft_logits,
            self.stats[3],
            self.stats[4],
            self.cu_num_logits,
            self.idx_mapping,
            self.temperature,
            self.seed,
            self.pos,
            self.cumulative_log_p,
            self.local_residual_mass,
            self.use_block_verification,
            token_logits=self.token_logits,
        )

    def resample_packet(self) -> torch.Tensor:
        """This shard's per-block resample winners, ``[2, num_reqs, num_blocks]``.

        Field 0 holds the global token id as integer bits, field 1 its
        Gumbel-perturbed value.
        """
        dtype = torch.float64 if self.use_fp64 else torch.float32
        num_blocks = triton.cdiv(self.shard.size, _RESAMPLE_BLOCK_SIZE)
        packet = self.target_logits.new_empty(2, self.num_reqs, num_blocks, dtype=dtype)
        _launch_resample(
            packet[0].view(_INT_VIEW_DTYPES[dtype]),
            packet[1],
            self.target_logits,
            self.target_rejected_logsumexp,
            self.draft_logits,
            self.draft_rejected_logsumexp,
            self.num_sampled,
            self.cu_num_logits,
            self.expanded_idx_mapping,
            self.draft_sampled,
            self.temperature,
            self.seed,
            self.pos,
            self.cumulative_log_p,
            self.shard.size,
            self.use_fp64,
            self.use_block_verification,
            token_offset=self.shard.start,
        )
        return packet

    def finish(self, gathered: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if gathered is None:
            assert not self.needs_resample
            # Greedy bonus rows got their target argmax with the stats.
            bonus_idx = self.cu_num_logits[1:] - 1
            resampled = self.stats[:2].view(torch.int32).index_select(1, bonus_idx)
            resampled_argmax = resampled[0]
            resampled_max = resampled[1].view(torch.float32)
        else:
            resampled = combine_vocab_blocks(gathered)
            resampled_argmax = resampled[0].view(_INT_VIEW_DTYPES[resampled.dtype])
            resampled_max = resampled[1]
        _launch_insert_resampled(
            self.sampled,
            self.num_sampled,
            resampled_argmax,
            resampled_max,
            self.cu_num_logits,
            self.expanded_idx_mapping,
            self.temperature,
        )
        return self.sampled, self.num_sampled


def rejection_sample_vocab_parallel(
    # [num_logits, shard.size], this rank's processed target logits
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, shard.size], this rank's q cache shard
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    shard: VocabShard,
    *,
    use_fp64: bool = False,
    use_block_verification: bool = False,
    all_greedy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rejection_sample`` over vocab-parallel logits, without gathering them.

    Takes the same inputs restricted to this rank's vocab shard and returns
    the same ``(sampled, num_sampled)`` on every rank. Synthetic acceptance
    rates and watermarking are unsupported. ``all_greedy`` must be set only
    when every request has temperature 0; it skips the exchanges that only
    sampled rows need.

    Exchanges at most three small packets: logits stats, residual mass
    partials (block verification with draft logits), and the per-block
    resample winners (not for all-greedy batches). Greedy tokens and every
    random draw match ``rejection_sample`` exactly; logsumexp and residual
    mass partials are summed in shard-sized groups, so sampled rows may differ
    only at floating-point near-ties.
    """
    sampler = VocabParallelRejection(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        shard,
        use_fp64=use_fp64,
        use_block_verification=use_block_verification,
        all_greedy=all_greedy,
    )
    sampler.apply_stats(shard.all_gather(sampler.stats_packet()))
    if sampler.needs_residual_mass:
        sampler.apply_residual_mass(shard.all_gather(sampler.residual_mass_packet()))
    sampler.reject()
    gathered = shard.all_gather(sampler.resample_packet()) if sampler.needs_resample else None
    return sampler.finish(gathered)
