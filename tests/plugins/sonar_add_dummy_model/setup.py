# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="sonar_add_dummy_model",
    version="0.1",
    packages=["sonar_add_dummy_model"],
    entry_points={"sonar.general_plugins": ["register_dummy_model = sonar_add_dummy_model:register"]},
)
