# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="aphrodite_add_dummy_model",
    version="0.1",
    packages=["aphrodite_add_dummy_model"],
    entry_points={"aphrodite.general_plugins": ["register_dummy_model = aphrodite_add_dummy_model:register"]},
)
