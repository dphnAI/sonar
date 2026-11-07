# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="sonar_add_dummy_endpoint_plugin",
    version="0.1",
    packages=["sonar_add_dummy_endpoint_plugin"],
    entry_points={
        "sonar.endpoint_plugins": [
            "dummy_admin_endpoint_plugin = sonar_add_dummy_endpoint_plugin:DummyAdminEndpointPlugin"
        ]
    },
)
