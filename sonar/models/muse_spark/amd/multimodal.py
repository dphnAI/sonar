# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark Omni for AMD ROCm GPUs: the shared Omni towers over the ROCm stack."""

from sonar.model_executor.models.interfaces import SupportsMultiModal
from sonar.models.muse_spark.common.omni import (
    OMNI_WEIGHTS_MAPPER,
    MuseSparkOmniMixin,
    register_omni_processor,
)

from .model import MuseSparkForCausalLM


@register_omni_processor
class MuseSparkForConditionalGeneration(MuseSparkOmniMixin, MuseSparkForCausalLM, SupportsMultiModal):
    """Muse Spark Omni: ROCm text stack plus NaViT vision and waveform audio."""

    hf_to_sonar_mapper = MuseSparkForCausalLM.hf_to_sonar_mapper | OMNI_WEIGHTS_MAPPER


__all__ = ["MuseSparkForConditionalGeneration"]
