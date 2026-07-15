# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight N-bit INT scheme with symmetric INT4 activation quant via Humming.

Handles compressed-tensors pack-quantized INT weight checkpoints (2-8 bit)
with INT4 symmetric dynamic per-token/per-group input activation
quantization. Static, per-tensor, and asymmetric activation quantization
are not supported.
"""

from aphrodite.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa8 import (  # noqa: E501
    CompressedTensorsWNA8Int,
)

__all__ = ["CompressedTensorsWNA4Int"]


class CompressedTensorsWNA4Int(CompressedTensorsWNA8Int):
    """INT4 activation variant of the Humming WNA8 scheme."""

    _kernel_backends_being_used: set[str] = set()
    _scheme_name = "WNA4Int"
