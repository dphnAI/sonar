# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from sonar.compilation.decorators import support_torch_compile
from sonar.config import (
    CompilationConfig,
    CompilationMode,
    SonarConfig,
    set_current_sonar_config,
)
from sonar.forward_context import set_forward_context
from sonar.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from sonar.sequence import IntermediateTensors


@support_torch_compile(
    dynamic_arg_dims={"x": 0, "deepstack_input_embeds": 0},
)
class _DeepStackDecoder(nn.Module):
    def __init__(self, *, sonar_config: SonarConfig, prefix: str = "") -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        deepstack_input_embeds: IntermediateTensors | None,
    ) -> torch.Tensor:
        if deepstack_input_embeds is not None:
            x = x + deepstack_input_embeds["deepstack_input_embeds_0"]
        return x


def _run_contract(
    deepstack_buffer: Qwen3VLForConditionalGeneration,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = SonarConfig(compilation_config=CompilationConfig(mode=CompilationMode.SONAR_COMPILE))
    with set_current_sonar_config(config):
        model = _DeepStackDecoder(sonar_config=config).eval().cuda()

    x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    with torch.inference_mode(), set_forward_context(None, sonar_config=config):
        empty_payload = deepstack_buffer._get_deepstack_input_embeds(num_tokens=2)
        first = model(x, empty_payload)

        payload = torch.full((1, 2, 4), 7.0, device="cuda")
        deepstack_buffer._set_deepstack_input_embeds(payload)
        active_payload = deepstack_buffer._get_deepstack_input_embeds(num_tokens=2)
        assert active_payload is not None
        second = model(x, active_payload)

    return x.cpu(), first.cpu(), second.cpu()


def test_qwen3_vl_deepstack_payload_after_empty_compile(monkeypatch):
    monkeypatch.setenv("SONAR_DISABLE_COMPILE_CACHE", "1")

    deepstack_buffer = object.__new__(Qwen3VLForConditionalGeneration)
    deepstack_buffer.deepstack_input_embeds = [torch.zeros(2, 4, device="cuda")]
    deepstack_buffer.deepstack_input_embeds_num_tokens = 0
    deepstack_buffer.deepstack_num_level = 1

    x, first, second = _run_contract(deepstack_buffer)
    torch.testing.assert_close(first, x)
    torch.testing.assert_close(second, x + 7)
