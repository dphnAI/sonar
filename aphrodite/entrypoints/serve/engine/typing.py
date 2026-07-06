# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol, TypeAlias

from aphrodite.config import ModelConfig
from aphrodite.entrypoints.chat_utils import ChatTemplateContentFormatOption
from aphrodite.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from aphrodite.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from aphrodite.entrypoints.openai.responses.protocol import ResponsesRequest
from aphrodite.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderChatRequest,
    DerenderCompletionRequest,
    GenerateRequest,
    GenerateResponse,
)
from aphrodite.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from aphrodite.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from aphrodite.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from aphrodite.renderers import ChatParams, TokenizeParams


class RendererRequest(Protocol):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        raise NotImplementedError


class RendererChatRequest(RendererRequest, Protocol):
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        raise NotImplementedError


CompletionLikeRequest: TypeAlias = (
    CompletionRequest | TokenizeCompletionRequest | DetokenizeRequest | DerenderCompletionRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest | BatchChatCompletionRequest | TokenizeChatRequest | DerenderChatRequest
)

SpeechToTextRequest: TypeAlias = TranscriptionRequest | TranslationRequest

AnyRequest: TypeAlias = (
    CompletionLikeRequest | ChatLikeRequest | SpeechToTextRequest | ResponsesRequest | GenerateRequest
)

AnyResponse: TypeAlias = (
    CompletionResponse | ChatCompletionResponse | TranscriptionResponse | TokenizeResponse | GenerateResponse
)
