# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project

import pytest
import torch

from aphrodite.multimodal.video_prune.vidcom2 import (
    compute_retained_tokens_count,
    compute_retention_mask,
)


def _fake_video_embeds(num_frames: int, rows: int, cols: int, hidden: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(num_frames * rows * cols, hidden, generator=generator)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("num_frames", [1, 4, 16])
def test_retention_mask(q: float, num_frames: int) -> None:
    merge = 2
    rows, cols = 6, 8
    mask = compute_retention_mask(
        _fake_video_embeds(num_frames, rows, cols),
        (num_frames, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=q,
    )
    expected = compute_retained_tokens_count(rows * cols, num_frames, q)
    assert mask.dtype == torch.bool
    assert mask.shape == (num_frames * rows * cols,)
    assert int(mask.sum()) == expected


def test_retained_count_floors_at_one_token_per_frame() -> None:
    """The global minimum is one token per frame (not a full first frame)."""
    assert compute_retained_tokens_count(tokens_per_frame=48, num_frames=4, q=0.999) == 4
    assert compute_retained_tokens_count(tokens_per_frame=48, num_frames=4, q=0.0) == 48 * 4


def test_mask_reads_frame_budgets_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original_tolist = torch.Tensor.tolist
    transfers: list[tuple[int, ...]] = []

    def reject_item(tensor: torch.Tensor) -> None:
        raise AssertionError(f"unexpected scalar read from shape {tuple(tensor.shape)}")

    def record_tolist(tensor: torch.Tensor) -> list[int]:
        transfers.append(tuple(tensor.shape))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "item", reject_item)
    monkeypatch.setattr(torch.Tensor, "tolist", record_tolist)
    embeds = _fake_video_embeds(num_frames=4, rows=6, cols=8)

    mask = compute_retention_mask(embeds, (4, 12, 16), spatial_merge_size=2, q=0.5)

    assert transfers == [(4,)]
    assert mask.shape == (4 * 6 * 8,)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("num_frames", [1, 4, 16])
def test_total_retained_matches_target(q: float, num_frames: int) -> None:
    """Mask total must equal the placeholder-sizing helper."""
    merge = 2
    rows, cols = 6, 8
    tpf = rows * cols
    embeds = _fake_video_embeds(num_frames, rows, cols)
    mask = compute_retention_mask(
        embeds,
        (num_frames, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=q,
    )
    expected = compute_retained_tokens_count(tokens_per_frame=tpf, num_frames=num_frames, q=q)
    assert int(mask.sum().item()) == expected


def test_per_frame_min_one_when_budget_allows() -> None:
    """No frame is fully dropped when the budget allows."""
    merge = 2
    rows, cols = 6, 8
    num_frames = 8
    embeds = _fake_video_embeds(num_frames, rows, cols)
    mask = compute_retention_mask(
        embeds,
        (num_frames, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=0.25,
    )
    per_frame = mask.view(num_frames, rows * cols).sum(dim=1)
    assert (per_frame >= 1).all(), f"zero-token frame detected: {per_frame.tolist()}"


def test_dynamic_per_frame_budget() -> None:
    """A distinctive frame gets more retained tokens than bland ones."""
    merge = 2
    rows, cols = 6, 8
    tpf = rows * cols
    hidden = 64
    torch.manual_seed(0)
    bland = 0.01 * torch.randn(tpf, hidden)
    frames = [torch.randn(tpf, hidden) * 1.0]
    for _ in range(7):
        frames.append(bland + 0.001 * torch.randn(tpf, hidden))
    embeds = torch.cat(frames, dim=0)
    mask = compute_retention_mask(
        embeds,
        (8, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=0.5,
    )
    per_frame = mask.view(8, tpf).sum(dim=1)
    assert per_frame[0].item() > per_frame[1:].float().mean().item()


def test_empty_input_safe() -> None:
    embeds = torch.zeros(0, 32)
    mask = compute_retention_mask(embeds, (0, 0, 0), 2, 0.25)
    assert mask.numel() == 0
