# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project

"""VidCom2 (Video Compression Commander) video token pruning."""

import torch
import torch.nn.functional as F

_ALPHAS: tuple[float, ...] = tuple(2.0**k for k in range(-3, 2))
_LOW_VAR_CHANNEL_RATIO = 0.5
_SOFTMAX_TEMPERATURE = 0.01


def compute_retained_tokens_count(tokens_per_frame: int, num_frames: int, q: float) -> int:
    """Return the number of video tokens retained after pruning."""
    total_tokens = tokens_per_frame * num_frames
    base_num = int(total_tokens * (1.0 - q))
    return max(num_frames, min(base_num, total_tokens))


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """Compute the VidCom2 retention mask for one video."""
    num_frames, height, width = map(int, video_size_thw)
    rows = height // spatial_merge_size
    cols = width // spatial_merge_size
    tokens_per_frame = rows * cols
    total_tokens = num_frames * tokens_per_frame
    device = video_embeds.device
    if tokens_per_frame == 0 or total_tokens == 0:
        return torch.ones(0, dtype=torch.bool, device=device)

    target = compute_retained_tokens_count(tokens_per_frame, num_frames, q)
    variances = video_embeds.var(dim=0, unbiased=False)
    num_channels = max(1, int(video_embeds.size(-1) * _LOW_VAR_CHANNEL_RATIO))
    low_var_idx = torch.topk(variances, k=num_channels, largest=False).indices
    selected = video_embeds.index_select(-1, low_var_idx)

    frames = F.normalize(selected.view(num_frames, tokens_per_frame, selected.size(-1)), dim=-1)
    video_center = frames.mean(dim=(0, 1), keepdim=True)
    frame_center = frames.mean(dim=1, keepdim=True)
    video_score = _multi_scale_gaussian(frames, video_center)
    frame_score = _multi_scale_gaussian(frames, frame_center)
    similarity = video_score + frame_score

    frame_distinctiveness = -video_score.mean(dim=-1)
    probabilities = F.softmax(
        (frame_distinctiveness - frame_distinctiveness.max()) / _SOFTMAX_TEMPERATURE,
        dim=0,
    )
    base = 1.0 - q
    scales = (base * (1.0 + probabilities - probabilities.mean())).clamp(max=1.0)
    budgets = (scales * tokens_per_frame).round().long().clamp(min=1, max=tokens_per_frame)

    mask = torch.zeros(num_frames, tokens_per_frame, dtype=torch.bool, device=device)
    for frame_idx in range(num_frames):
        keep = int(budgets[frame_idx].item())
        indices = torch.topk(similarity[frame_idx], k=keep, largest=False, sorted=False).indices
        mask[frame_idx].scatter_(0, indices, True)

    flat_mask = mask.view(-1)
    flat_similarity = similarity.view(-1)
    current = int(flat_mask.sum().item())
    if current > target:
        retained = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        worst = torch.topk(
            flat_similarity[retained],
            k=current - target,
            largest=True,
            sorted=False,
        ).indices
        flat_mask[retained[worst]] = False
    elif current < target:
        available = (~flat_mask).nonzero(as_tuple=False).squeeze(-1)
        best = torch.topk(
            flat_similarity[available],
            k=target - current,
            largest=False,
            sorted=False,
        ).indices
        flat_mask[available[best]] = True
    return flat_mask


def _multi_scale_gaussian(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2.0 * alpha)) for alpha in _ALPHAS)
