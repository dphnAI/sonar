# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from sonar.parser.engine.registered_adapters import RippleParserToolAdapter


class RippleEngineToolParser(RippleParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "hermes"
