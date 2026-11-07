# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def use_tensor_descriptor(override: bool | None = None) -> bool:
    """Tri-state TD setting: unset=auto (on for XPU), 1/0=force on/off."""
    from aphrodite import envs
    from aphrodite.platforms import current_platform

    if override is None:
        override = envs.APHRODITE_TRITON_USE_TD
    if override is not None:
        return override
    return current_platform.is_xpu()
