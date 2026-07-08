# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.transformers_utils.configs.speculators.base import SpeculatorsConfig


def _dflash_config(**overrides):
    config = {
        "speculators_model_type": "dflash",
        "transformer_layer_config": {
            "model_type": "qwen3",
        },
        "draft_vocab_size": 42,
        "target_hidden_size": 16,
        "mask_token_id": 99,
        "aux_hidden_state_layer_ids": [2, 4],
    }
    config.update(overrides)
    return config


def test_dflash_speculators_config_defaults_to_non_causal_swa():
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(_dflash_config())

    assert config["dflash_config"] == {
        "mask_token_id": 99,
        "target_layer_ids": [1, 3],
        "causal": False,
    }


def test_dflash_speculators_config_enables_causal_swa():
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(_dflash_config(sliding_window_non_causal=False))

    assert config["dflash_config"] == {
        "mask_token_id": 99,
        "target_layer_ids": [1, 3],
        "causal": True,
    }
