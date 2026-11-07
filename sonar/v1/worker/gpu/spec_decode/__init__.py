# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

from aphrodite.utils.import_utils import resolve_obj_by_qualname

# Keep this light: aphrodite.config imports the speculator configs in this package.
if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig
    from aphrodite.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
    from aphrodite.v1.worker.gpu.states import RequestState


def init_speculator(
    aphrodite_config: "AphroditeConfig",
    device: torch.device,
    req_states: "RequestState",
):
    """Build the speculator named by `speculative_config.speculator_config`.

    The speculator is constructed with `(aphrodite_config, device)`, plus
    `req_states` if its constructor accepts it.
    """
    from aphrodite.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    speculator_config = speculative_config.speculator_config
    qualname = speculator_config.speculator_class
    if qualname is None:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
    speculator_cls = resolve_obj_by_qualname(qualname)
    if not (isinstance(speculator_cls, type) and issubclass(speculator_cls, BaseSpeculator)):
        raise TypeError(
            f"speculator_class '{qualname}' of {type(speculator_config).__name__} "
            f"must be a subclass of {BaseSpeculator.__module__}."
            f"{BaseSpeculator.__qualname__}."
        )
    # BaseSpeculator doesn't define the constructor signature.
    factory = cast(Callable[..., "BaseSpeculator"], speculator_cls)
    if "req_states" in inspect.signature(factory).parameters:
        return factory(aphrodite_config, device, req_states=req_states)
    return factory(aphrodite_config, device)
