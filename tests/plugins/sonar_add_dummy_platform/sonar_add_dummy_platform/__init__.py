# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def dummy_platform_plugin() -> str | None:
    # Real platform plugins often import Sonar modules while registering.
    from sonar.utils.torch_utils import direct_register_custom_op  # noqa: F401

    return "sonar_add_dummy_platform.dummy_platform.DummyPlatform"


def register_ops():
    import sonar_add_dummy_platform.dummy_custom_ops  # noqa
