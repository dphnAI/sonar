# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="aphrodite_add_dummy_platform",
    version="0.1",
    packages=["aphrodite_add_dummy_platform"],
    entry_points={
        "aphrodite.platform_plugins": [
            "dummy_platform_plugin = aphrodite_add_dummy_platform:dummy_platform_plugin"  # noqa
        ],
        "aphrodite.general_plugins": [
            "dummy_custom_ops = aphrodite_add_dummy_platform:register_ops"
        ],
    },
)
