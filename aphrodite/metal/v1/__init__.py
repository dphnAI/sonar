# SPDX-License-Identifier: Apache-2.0
"""Aphrodite v1 compatibility module for Metal platform."""

__all__ = ["MetalWorker"]


def __getattr__(name: str):
    if name == "MetalWorker":
        from aphrodite.v1.worker.metal_worker import MetalWorker

        return MetalWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
