# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling over tensor-parallel vocabulary shards without gathering logits.

Each rank keeps its own ``[start, start + size)`` slice of the vocabulary and
exchanges only small per-row partial results. Every random draw is keyed by
the global token id, so sampling a shard reproduces the per-token noise of the
full-vocabulary path.
"""

from dataclasses import dataclass

import torch

from aphrodite.distributed import get_tp_group


@dataclass(frozen=True)
class VocabShard:
    """One rank's contiguous, unpadded slice of the vocabulary."""

    rank: int
    world_size: int
    start: int
    size: int
    vocab_size: int

    @property
    def end(self) -> int:
        return self.start + self.size

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Stack every rank's ``tensor`` along a new leading dimension.

        Ranks must pass tensors of identical shape and dtype. The result is
        ordered by rank, which is also ascending vocabulary order.
        """
        flat = tensor.contiguous().view(1, -1)
        gathered = get_tp_group().all_gather(flat, dim=0)
        return gathered.view(self.world_size, *tensor.shape)

    def gather_rows(self, local: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Rebuild full-vocab rows of a sharded ``[R, ..., size]`` cache.

        Writes ``out[rows]`` from every rank's ``local[rows]`` and returns
        ``out``. Rows not listed in ``rows`` are left untouched.
        """
        selected = local[rows]
        gathered = self.all_gather(selected)
        # [world, n, ..., size] -> [n, ..., world * size] in vocab order.
        full = gathered.movedim(0, -2).reshape(*selected.shape[:-1], self.vocab_size)
        out[rows] = full
        return out


def vocab_shard_for(lm_head: torch.nn.Module) -> VocabShard | None:
    """Return the shard of a vocab-parallel head, or None if it is unsuitable.

    Shards must be equal-sized, contiguous and free of padding or added
    vocabulary, so a local column ``i`` is global token ``start + i``.
    """
    tp_size = getattr(lm_head, "tp_size", 1)
    if tp_size <= 1:
        return None
    indices = lm_head.shard_indices
    vocab_size = lm_head.org_vocab_size
    size = lm_head.num_embeddings_per_partition
    if (
        lm_head.num_added_embeddings != 0
        or lm_head.num_embeddings_padded != vocab_size
        or size * tp_size != vocab_size
        or indices.num_org_vocab_padding != 0
        or indices.org_vocab_start_index != lm_head.tp_rank * size
        or indices.org_vocab_end_index != indices.org_vocab_start_index + size
    ):
        return None
    return VocabShard(
        rank=lm_head.tp_rank,
        world_size=tp_size,
        start=indices.org_vocab_start_index,
        size=size,
        vocab_size=vocab_size,
    )
