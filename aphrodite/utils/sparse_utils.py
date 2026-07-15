# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread-safe helpers for sparse tensor invariant validation."""

import contextlib
import threading
from collections.abc import Iterator

import torch

_SPARSE_LOAD_LOCK = threading.Lock()


@contextlib.contextmanager
def check_sparse_tensor_invariants_threadsafe() -> Iterator[None]:
    """Serialize PyTorch's process-global sparse invariant flag."""
    with _SPARSE_LOAD_LOCK, torch.sparse.check_sparse_tensor_invariants():
        yield
