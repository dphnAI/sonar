# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility fix for CuTe's equality-based missing-argument check.

The supported binder compares tensor arguments with a private sentinel while
launching eager kernels. Identity is the required test; tensor equality can
raise internal C++ exceptions. Keep every other argument-binding rule unchanged.
"""

import ast
import hashlib
import inspect
import textwrap
from functools import cache
from typing import Any

from cutlass.base_dsl.common import DSLUserCodeError
from cutlass.base_dsl.diagnostics import DiagId

from aphrodite.logger import init_logger

logger = init_logger(__name__)
_SUPPORTED_BINDER = "ced87f332007db7fcacccd1e3f93d1ff2dd621b247958e74a81eb0eb07d7c130"


@cache
def install_identity_argument_check() -> None:
    """Patch only the verified vendor implementation; leave other versions alone."""
    from cutlass.base_dsl.jit_executor import ExecutionArgs

    original = ExecutionArgs.get_rectified_args
    if original is _get_rectified_args:
        return
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError):
        logger.warning_once("Cannot identify CuTe argument binder; leaving it unchanged.")
        return
    digest = hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()
    if digest != _SUPPORTED_BINDER:
        logger.warning_once("CuTe argument binder changed; leaving it unchanged.")
        return
    ExecutionArgs.get_rectified_args = _get_rectified_args
    logger.info_once("CuTe kernel arguments use identity-only missing-argument checks.")


def _get_rectified_args(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    meta = self._meta
    missing = self._missing
    pos_count = len(meta.pos_names)
    if len(args) > pos_count:
        raise DSLUserCodeError(DiagId.CALL_TOO_MANY_ARGS, expected=pos_count, provided=len(args))
    rectified = [missing] * meta.arg_count
    pos_len = len(args)
    rectified[:pos_len] = args
    for i in range(pos_len, pos_count):
        default = meta.pos_defaults[i]
        if default is not missing:
            rectified[i] = default
    for j, default in enumerate(meta.kwonly_defaults):
        if default is not missing:
            rectified[pos_count + j] = default
    for name, value in kwargs.items():
        index = meta.name_to_index.get(name)
        if index is None:
            raise DSLUserCodeError(DiagId.CALL_UNEXPECTED_KWARG, argument_name=name)
        if index < pos_len:
            raise DSLUserCodeError(DiagId.CALL_DUPLICATE_ARGUMENT, argument_name=name)
        rectified[index] = value
    if any(value is missing for value in rectified):
        raise DSLUserCodeError(
            DiagId.CALL_MISSING_ARGS,
            missing=[name for name, value in zip(meta.all_names, rectified) if value is missing],
        )
    return rectified
