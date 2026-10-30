# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel argument binding must not invoke user/tensor equality."""

import inspect

import pytest
import torch

pytest.importorskip("cutlass.base_dsl.jit_executor")

from cutlass.base_dsl.common import DSLUserCodeError  # noqa: E402
from cutlass.base_dsl.diagnostics import DiagId  # noqa: E402
from cutlass.base_dsl.jit_executor import ExecutionArgs  # noqa: E402

from aphrodite.models.deepseek_v41.sm120.launch_args import _get_rectified_args  # noqa: E402


def kernel(x, scale=2, *, out, stream=None):
    pass


class EqualityForbidden:
    def __eq__(self, other):
        raise AssertionError("Binding must not compare runtime argument values")


def test_tensor_values_and_defaults():
    binder = ExecutionArgs(inspect.signature(kernel), "kernel")
    x = torch.ones(2)
    out = EqualityForbidden()
    result = _get_rectified_args(binder, (x,), {"out": out})
    assert result[0] is x
    assert result[1] == 2
    assert result[2] is out
    assert result[3] is None
    explicit = _get_rectified_args(binder, (), {"x": x, "scale": 3, "out": out, "stream": out})
    assert explicit[0] is x and explicit[1] == 3
    assert explicit[2] is out and explicit[3] is out


@pytest.mark.parametrize(
    ("args", "kwargs", "diagnostic"),
    [
        ((1, 2, 3), {"out": 4}, DiagId.CALL_TOO_MANY_ARGS),
        ((1,), {"x": 2, "out": 4}, DiagId.CALL_DUPLICATE_ARGUMENT),
        ((1,), {"out": 4, "unknown": 5}, DiagId.CALL_UNEXPECTED_KWARG),
        ((), {"out": 4}, DiagId.CALL_MISSING_ARGS),
        ((1,), {}, DiagId.CALL_MISSING_ARGS),
    ],
)
def test_invalid_bindings_report_diagnostic(args, kwargs, diagnostic):
    binder = ExecutionArgs(inspect.signature(kernel), "kernel")
    with pytest.raises(DSLUserCodeError) as error:
        _get_rectified_args(binder, args, kwargs)
    assert error.value.diag_id is diagnostic
