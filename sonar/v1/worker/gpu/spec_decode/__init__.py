# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

from sonar.utils.import_utils import resolve_obj_by_qualname

# Keep this light: sonar.config imports the speculator configs in this package.
if TYPE_CHECKING:
    from sonar.config import SonarConfig
    from sonar.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
    from sonar.v1.worker.gpu.states import RequestState


def init_speculator(
    sonar_config: "SonarConfig",
    device: torch.device,
    req_states: "RequestState",
):
    """Build the speculator named by `speculative_config.speculator_config`.

    The speculator is constructed with `(sonar_config, device)`, plus
    `req_states` if its constructor accepts it.
    """
    from sonar.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

    speculative_config = sonar_config.speculative_config
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
        return factory(sonar_config, device, req_states=req_states)
    return factory(sonar_config, device)
