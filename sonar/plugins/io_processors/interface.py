# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Generic, TypeVar

from aphrodite.config import AphroditeConfig
from aphrodite.inputs import PromptType
from aphrodite.outputs import PoolingRequestOutput
from aphrodite.pooling_params import PoolingParams
from aphrodite.renderers import BaseRenderer
from aphrodite.sampling_params import SamplingParams

IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """Abstract interface for pre/post-processing of engine I/O."""

    def __init__(self, aphrodite_config: AphroditeConfig, renderer: BaseRenderer):
        super().__init__()

        self.aphrodite_config = aphrodite_config

    def parse_data(self, data: object) -> IOProcessorInput:
        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        return params or SamplingParams()

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        return params or PoolingParams(task="plugin")

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # We cannot guarantee outputs are returned in the same order they were
        # fed to Aphrodite.
        # Let's sort them by id before post_processing
        sorted_output = sorted([(i, item) async for i, item in model_output], key=lambda output: output[0])
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id=request_id, **kwargs)
