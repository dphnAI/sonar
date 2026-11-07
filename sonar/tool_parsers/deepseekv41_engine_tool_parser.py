# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from sonar.parser.engine.registered_adapters import DeepSeekV41ParserToolAdapter


class DeepSeekV41EngineToolParser(DeepSeekV41ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "deepseek_v4_1"
