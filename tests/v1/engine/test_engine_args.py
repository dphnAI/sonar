# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError

import pytest

from aphrodite.config import AphroditeConfig
from aphrodite.engine.arg_utils import EngineArgs
from aphrodite.usage.usage_lib import UsageContext
from aphrodite.utils.argparse_utils import FlexibleArgumentParser
from aphrodite.utils.hashing import _xxhash


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    aphrodite_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert aphrodite_config.cache_config.enable_prefix_caching, (
        "V1 turns on prefix caching by default."
    )

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    aphrodite_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not aphrodite_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    aphrodite_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert aphrodite_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    aphrodite_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    aphrodite_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    aphrodite_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    aphrodite_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert aphrodite_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    aphrodite_config: AphroditeConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)

    from aphrodite.platforms import current_platform
    from aphrodite.utils.mem_constants import GiB_bytes

    device_memory = current_platform.get_device_total_memory()
    device_name = current_platform.get_device_name().lower()
    if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
        # For GPUs like H100, H200, and MI300x with >= 70GB memory
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256

    assert aphrodite_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert aphrodite_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    aphrodite_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert aphrodite_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert aphrodite_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501


def test_mm_prefix_lm_raises_batched_tokens_floor():
    """Verify that prefix-LM multimodal models auto-raise
    max_num_batched_tokens to fit at least one multimodal item.

    Regression test for https://github.com/vllm-project/vllm/issues/42687
    """
    from unittest.mock import patch

    # Simulate a prefix-LM multimodal model whose largest modality
    # (video) requires 2496 tokens — more than the 2048 default.
    fake_mm_min = (2496, "video")

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_model_len=2048,
        enforce_eager=True,
    )

    with (
        patch.object(
            type(engine_args),
            "_get_min_mm_batched_tokens",
            staticmethod(lambda _mc: fake_mm_min),
        ),
        patch(
            "aphrodite.config.ModelConfig.is_multimodal_model",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "aphrodite.config.ModelConfig.is_mm_prefix_lm",
            new_callable=lambda: property(lambda self: True),
        ),
    ):
        aphrodite_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert aphrodite_config.scheduler_config.max_num_batched_tokens >= 2496
