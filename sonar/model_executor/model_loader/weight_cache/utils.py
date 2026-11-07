# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the weight cache daemon, the IPC loader and the engine.

These live outside ``protocol`` so callers that only need to know whether a
draft is cached, or how a daemon group is named, do not have to import the
wire format.
"""

from typing import TYPE_CHECKING, Any, cast

import torch

from aphrodite.config import AphroditeConfig, SpeculativeConfig, replace, set_current_aphrodite_config
from aphrodite.model_executor.models.interfaces import SupportsEagleBase
from aphrodite.platforms import current_platform

if TYPE_CHECKING:
    from aphrodite.config import ModelConfig
    from aphrodite.v1.worker.gpu_model_runner import GPUModelRunner

# Speculative methods whose draft model the daemon caches in its own group.
# Other drafts keep loading from disk in the engine.
WEIGHT_CACHE_DRAFT_METHODS = frozenset({"mtp", "eagle", "eagle3"})

# Python-side flags that weight loading sets on EAGLE-style drafts; the engine
# never runs load_weights for cached models, so the daemon ships them.
EXPORTED_MODEL_ATTRS = tuple(SupportsEagleBase.__annotations__)


def export_model_attrs(model: Any) -> dict[str, bool]:
    return {name: bool(getattr(model, name)) for name in EXPORTED_MODEL_ATTRS if hasattr(model, name)}


def is_draft_model_cacheable(speculative_config: SpeculativeConfig | None) -> bool:
    """Whether the daemon serves the speculative draft as a separate role."""
    return (
        speculative_config is not None
        and speculative_config.method in WEIGHT_CACHE_DRAFT_METHODS
        and speculative_config.draft_model_config is not None
    )


def format_daemon_role(is_draft: bool) -> str:
    """Name of a daemon group: the target model or the draft."""
    return "draft" if is_draft else "target"


def format_socket_role_suffix(is_draft: bool) -> str:
    """Socket-name suffix keeping the draft group distinct from the target."""
    return "_draft" if is_draft else ""


def build_warmup_runner(
    aphrodite_config: AphroditeConfig,
    local_rank: int,
    *,
    is_draft: bool,
    model_config: "ModelConfig",
) -> "GPUModelRunner":
    """Create a model runner for the daemon's warmup dummy runs."""
    if is_draft:
        aphrodite_config = replace(aphrodite_config, model_config=model_config, speculative_config=None)
    device = torch.device(current_platform.device_type, local_rank)
    with set_current_aphrodite_config(aphrodite_config):
        if aphrodite_config.use_v2_model_runner:
            from aphrodite.v1.worker.gpu.model_runner import GPUModelRunner as Runner
        else:
            from aphrodite.v1.worker.gpu_model_runner import GPUModelRunner as Runner
        runner = Runner(aphrodite_config, device)
    # The two runner classes share the warmup surface; the nominal type
    # stays V1's GPUModelRunner.
    return cast("GPUModelRunner", runner)
