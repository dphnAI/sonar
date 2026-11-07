# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder output widths for sizing EC transfers.

An encoder can emit features wider than the LM embedding (DeepStack packs
several layers on the last dim) by an amount only the model knows. Producers
size transfers before the first real item is encoded, so the engine measures
the width once at startup by encoding a dummy item per modality.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from aphrodite.logger import init_logger

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig

logger = init_logger(__name__)


@torch.inference_mode()
def _measure_on_worker(worker: Any) -> dict[str, int]:
    """Encode one dummy item per modality; runs on every worker.

    Returns {} on ranks that hold no encoder.
    """
    from aphrodite.distributed.parallel_state import get_pp_group
    from aphrodite.multimodal import MULTIMODAL_REGISTRY
    from aphrodite.multimodal.encoder_budget import MultiModalBudget
    from aphrodite.multimodal.utils import group_and_batch_mm_kwargs
    from aphrodite.utils.torch_utils import PIN_MEMORY

    runner = worker.model_runner
    if not runner.supports_mm_inputs or not get_pp_group().is_first_rank:
        return {}
    budget = MultiModalBudget(worker.aphrodite_config, MULTIMODAL_REGISTRY, enable_cache=False)
    model = runner.get_model()
    widths: dict[str, int] = {}
    for modality in budget.mm_max_toks_per_item:
        items = budget.get_dummy_encoder_profile_inputs(modality, 1)
        if not items:
            continue
        _, _, mm_kwargs = next(group_and_batch_mm_kwargs(items, device=runner.device, pin_memory=PIN_MEMORY))
        outputs = model.embed_multimodal(**mm_kwargs)
        widths[modality] = int(outputs[0].shape[-1])
        del outputs
    return widths


def measure_encoder_output_widths(
    aphrodite_config: "AphroditeConfig", collective_rpc: Callable[..., list[Any]]
) -> None:
    """Record the encoder output widths on an EC producer's config.

    Call after the model is loaded and before the KV cache is allocated: the
    Scheduler's connector reads the widths, and once the KV cache holds the
    memory budget there is no room left for a dummy encode.

    Raises:
        ValueError: No worker could encode a dummy item, or workers disagree.

    """
    ec_config = aphrodite_config.ec_transfer_config
    if ec_config is None or not ec_config.is_ec_producer:
        return
    reports = [widths for widths in collective_rpc(_measure_on_worker) if widths]
    if not reports:
        raise ValueError(
            "EC producer could not measure the encoder output width: the model encoded no dummy multimodal input."
        )
    if any(widths != reports[0] for widths in reports):
        raise ValueError(f"Workers disagree on the encoder output widths: {reports}")
    logger.info("EC producer encoder output widths: %s", reports[0])
    ec_config.mm_encoder_output_widths = reports[0]


def get_encoder_output_width(aphrodite_config: "AphroditeConfig", modality: str) -> int:
    """Encoder output width for `modality`, as measured at producer startup.

    Raises:
        ValueError: No width was measured for `modality`, so a transfer
            cannot be sized.

    """
    assert aphrodite_config.ec_transfer_config is not None
    widths = aphrodite_config.ec_transfer_config.mm_encoder_output_widths
    if modality not in widths:
        raise ValueError(f"No encoder output width was measured for modality {modality!r}; measured: {sorted(widths)}")
    return widths[modality]
