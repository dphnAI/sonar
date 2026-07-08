# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``aphrodite.v1.worker.xpu_model_runner`` (XPU worker / CUDA shims)."""

import pytest
import torch
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

from aphrodite.v1.worker.xpu_model_runner import _torch_cuda_wrapper

# XPU-only: needs distinct torch.cuda vs torch.xpu current_stream symbols.
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not hasattr(torch.xpu, "current_stream"),
    reason="torch.xpu.current_stream is required",
)


# Child process: patched torch.cuda must not leak to other tests in the session.
@pytest.mark.forked
def test_torch_cuda_wrapper_allows_dynamo_handler_registration() -> None:
    """Guard against XPU CUDA shim breaking Torch Dynamo during AOT compile.

    Before the fix, ``_torch_cuda_wrapper`` assigned
    ``torch.cuda.current_stream = torch.xpu.current_stream`` (same function object).
    On the first AOT/profile run, Dynamo builds its in-graph handler table and
    registers ``torch.cuda.current_stream`` and ``torch.xpu.current_stream``
    separately; duplicate identity triggers::

        AssertionError: Handler already registered for <function current_stream ...>

    That surfaced as EngineCore failing in ``profile_run`` / ``_get_handlers()``.
    The fix uses distinct shim callables so both can be registered.

    This test replays the post-init state (wrapper applied, patches left on
    ``torch.cuda``) and checks that Dynamo's real ``_get_handlers()`` succeeds.
    """
    # Same entry point as XPUModelRunner.__init__ (patches persist after exit).
    with _torch_cuda_wrapper():
        pass

    # Fresh handler table build, as on first torch.compile / AOT in the worker.
    # Registers torch.cuda.current_stream and torch.xpu.current_stream separately;
    # if they are the same object (pre-fix alias), raises Handler already registered.
    TorchInGraphFunctionVariable._get_handlers.cache_clear()
    TorchInGraphFunctionVariable._get_handlers()


@pytest.mark.forked
def test_torch_cuda_wrapper_drops_event_blocking_kwarg(monkeypatch) -> None:
    event_calls: list[dict[str, object]] = []

    class FakeXPUEvent:
        def __init__(self, *args, **kwargs) -> None:
            event_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(torch.xpu, "Event", FakeXPUEvent)

    with _torch_cuda_wrapper():
        event = torch.cuda.Event(
            enable_timing=True,
            blocking=True,
            interprocess=False,
        )

    assert isinstance(event, FakeXPUEvent)
    assert event_calls == [
        {
            "args": (),
            "kwargs": {
                "enable_timing": True,
                "interprocess": False,
            },
        }
    ]
