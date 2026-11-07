# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.parser.engine.registered_adapters import MinimaxM3ParserToolAdapter
from aphrodite.tool_parsers.rust_tool_parser import RustToolParser


class MinimaxM3ToolParser(MinimaxM3ParserToolAdapter):  # type: ignore[valid-type, misc]
    # Required and named tool choices parse the native M3 syntax instead of
    # forcing JSON output.
    supports_required_and_named = False


class MinimaxM3RustToolParser(RustToolParser):
    """Select with ``--tool-call-parser minimax_m3_rust``.

    This uses Sonar's optional PyO3 parser extension; ``minimax_m3`` selects
    the Python parser engine.
    """

    rust_parser_name = "MinimaxM3ToolParser"
    tool_call_start_token = "]<]minimax[>[<tool_call>"
