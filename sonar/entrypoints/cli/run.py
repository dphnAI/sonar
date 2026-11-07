# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.entrypoints.cli.serve import (
    ServeSubcommand,
)


class RunSubcommand(ServeSubcommand):
    """Compatibility alias for the legacy `run` CLI."""

    name = "run"


def cmd_init():
    return [RunSubcommand()]
