# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="aphrodite_add_dummy_endpoint_plugin",
    version="0.1",
    packages=["aphrodite_add_dummy_endpoint_plugin"],
    entry_points={
        "aphrodite.endpoint_plugins": [
            "dummy_admin_endpoint_plugin = aphrodite_add_dummy_endpoint_plugin:DummyAdminEndpointPlugin"
        ]
    },
)
