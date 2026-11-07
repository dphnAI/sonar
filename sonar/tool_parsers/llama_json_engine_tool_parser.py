# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.parser.engine.registered_adapters import LlamaJsonParserToolAdapter


class LlamaJsonEngineToolParser(LlamaJsonParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "llama"
