# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Description: Test the lazy import module
# The utility function cannot be placed in `sonar.utils`
# this needs to be a standalone script
import sys
from unittest.mock import patch

# List of modules that should not be imported too early.
# Lazy import `torch._inductor.async_compile` to avoid creating
# too many processes before we set the number of compiler threads.
# Lazy import `cv2` to avoid bothering users who only use text models.
# `cv2` can easily mess up the environment.
# `openai`, `fastapi` and `starlette` belong to the frontend layer; `sonar.config`
# (imported by every process) and the engine core's `sonar.v1.structured_output`
# must not pull them in.
module_names = [
    "torch._inductor.async_compile",
    "cv2",
    "openai",
    "fastapi",
    "starlette",
]

# set all modules in `module_names` to be None.
# if we import any modules during `import sonar`, there would be a
# hard error and nice stacktrace on the first import.
with patch.dict(sys.modules, dict.fromkeys(module_names)):
    import sonar  # noqa: F401
    import sonar.config  # noqa: F401
    import sonar.v1.structured_output  # noqa: F401
