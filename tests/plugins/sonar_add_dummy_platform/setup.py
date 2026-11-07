# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="sonar_add_dummy_platform",
    version="0.1",
    packages=["sonar_add_dummy_platform"],
    entry_points={
        "sonar.platform_plugins": [
            "dummy_platform_plugin = sonar_add_dummy_platform:dummy_platform_plugin"  # noqa
        ],
        "sonar.general_plugins": ["dummy_custom_ops = sonar_add_dummy_platform:register_ops"],
    },
)
