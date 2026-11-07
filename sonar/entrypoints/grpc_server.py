# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings

from sonar.entrypoints.launchers.grpc_server import main, serve_grpc

warnings.warn(
    "`sonar.entrypoints.grpc_server is deprecated and will likely be "
    "unsupported in a future version. Use the corresponding function from "
    "`sonar.entrypoints.launchers.grpc_server` instead.",
    DeprecationWarning,
    stacklevel=1,
)

__all__ = ["main", "serve_grpc"]


if __name__ == "__main__":
    warnings.warn(
        "`python -m sonar.entrypoints.grpc_server is deprecated and "
        "will likely be unsupported in a future version. "
        "Please use `sonar serve <model_path> --grpc` instead.",
        DeprecationWarning,
        stacklevel=1,
    )

    main()
