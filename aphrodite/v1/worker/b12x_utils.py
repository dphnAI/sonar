# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x-specific adapters for worker profiling and MRV2 graph capture."""

import gc
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from aphrodite.v1.worker.gpu.model_runner import GPUModelRunner
    from aphrodite.v1.worker.gpu_worker import Worker


def prepare_b12x_weights(worker: "Worker") -> None:
    from aphrodite.model_executor.warmup.b12x_prepare import (
        b12x_native_supported,
        run_b12x_preparation,
    )

    if b12x_native_supported(worker):
        run_b12x_preparation(worker, stage="weights")
        # Timing trials must not distort subsequent KV-cache admission.
        gc.collect()
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()


def profile_b12x_memory(worker: "Worker", *, cudagraphs: bool = False):
    from aphrodite.model_executor.warmup.b12x_prepare import (
        b12x_native_supported,
        run_b12x_preparation,
    )

    native = b12x_native_supported(worker)
    batch = None

    def prepare_profile_state() -> None:
        nonlocal batch
        batch = run_b12x_preparation(worker, stage="state", autotune=False)

    try:
        if cudagraphs:
            if native:
                return worker.model_runner.profile_cudagraph_memory(prepare_profile_state)
            return worker.model_runner.profile_cudagraph_memory()
        if native:
            return cast("GPUModelRunner", worker.model_runner).profile_run(
                randomize_inputs=worker.randomize_dummy_inputs,
                prepare_profile_state=prepare_profile_state,
            )
        return worker.model_runner.profile_run(randomize_inputs=worker.randomize_dummy_inputs)
    finally:
        if batch is not None:
            # The runner destroys profiling graphs/cache aliases before their
            # plan ownership is released, including exceptional exits.
            batch.release()
            gc.collect()
            torch.accelerator.empty_cache()


def capture_b12x_model(worker: "Worker") -> int:
    if worker._b12x_session is not None:
        return worker.model_runner.capture_model(preparation_session=worker._b12x_session)
    return worker.model_runner.capture_model()


def warmup_b12x_cudagraphs(runner: "GPUModelRunner", *, capture_decoder: bool) -> None:
    from aphrodite.model_executor.warmup.b12x_warmup import warmup_b12x_model
    from aphrodite.v1.worker.gpu.cudagraph_utils import get_speculator_cudagraph_managers

    manager = runner.cudagraph_manager
    assert manager is not None
    draft_managers = get_speculator_cudagraph_managers(runner.speculator)
    if capture_decoder:
        warmup_b12x_model(runner.model, tuple(manager.planned_token_counts()), runner.dtype)
        draft_model = runner.get_draft_model()
        if draft_model is not None:
            draft_counts = {count for manager in draft_managers.values() for count in manager.planned_token_counts()}
            warmup_b12x_model(draft_model, tuple(sorted(draft_counts)), runner.dtype)
    if getattr(runner.model, "supports_b12x_native_preparation", False):
        manager.synchronize_warmup = True
        for draft_manager in draft_managers.values():
            draft_manager.synchronize_warmup = True
