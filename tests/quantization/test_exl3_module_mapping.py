# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from sonar.model_executor.layers.quantization.exl3 import Exl3Config
from sonar.model_executor.models.utils import WeightsMapper


def test_exl3_metadata_follows_model_module_renames():
    storage = {
        "model.layers.0.conv.in_proj": {"quant_format": "exl3"},
        "model.layers.0.feed_forward.w1": {"quant_format": "exl3"},
        "model.layers.0.feed_forward.w3": {"quant_format": "exl3"},
        "model.layers.0.conv.conv": {"quant_format": "fp16"},
    }
    config = Exl3Config(tensor_storage=storage)
    mapper = WeightsMapper(orig_to_new_substr={".conv.": ".short_conv."})

    config.apply_sonar_mapper(mapper)

    assert config._linear_prefix_is_exl3("model.layers.0.short_conv.in_proj")
    assert config._linear_prefix_is_exl3("model.layers.0.feed_forward.w13")
    assert not config._linear_prefix_is_exl3("model.layers.0.short_conv.conv")
    assert not config._linear_prefix_is_exl3("model.layers.0.conv.in_proj")
    assert "model.layers.0.conv.in_proj" in storage
