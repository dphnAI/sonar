"""Schemas and utilities for tokenization inputs."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias, TypedDict

from sonar.inputs import EmbedsPrompt, TokensPrompt

DecoderOnlyTokPrompt: TypeAlias = TokensPrompt | EmbedsPrompt
"""
A [`DecoderOnlyDictPrompt`][sonar.renderers.inputs.preprocess.DecoderOnlyDictPrompt]
that has been tokenized.
"""


EncoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`EncoderDictPrompt`][sonar.renderers.inputs.preprocess.EncoderDictPrompt]
that has been tokenized.
"""


DecoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`DecoderDictPrompt`][sonar.renderers.inputs.preprocess.DecoderDictPrompt]
that has been tokenized.
"""


class EncoderDecoderTokPrompt(TypedDict):
    """A
    [`EncoderDecoderDictPrompt`][sonar.renderers.inputs.preprocess.EncoderDecoderDictPrompt]
    that has been tokenized.
    """

    encoder_prompt: EncoderTokPrompt

    decoder_prompt: DecoderTokPrompt | None


SingletonTokPrompt: TypeAlias = DecoderOnlyTokPrompt | EncoderTokPrompt | DecoderTokPrompt
"""
A [`SingletonDictPrompt`][sonar.renderers.inputs.preprocess.SingletonDictPrompt]
that has been tokenized.
"""


TokPrompt: TypeAlias = DecoderOnlyTokPrompt | EncoderDecoderTokPrompt
"""
A [`DictPrompt`][sonar.renderers.inputs.preprocess.DictPrompt]
that has been tokenized.
"""
