# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import openai
import pytest

from tests.conftest import HfRunner
from tests.models.utils import EmbedModelInfo, check_embeddings_close, matryoshka_fy


def run_embedding_correctness_test(
    hf_model: "HfRunner",
    inputs: list[str],
    aphrodite_outputs: Sequence[list[float]],
    dimensions: int | None = None,
):
    hf_outputs = hf_model.encode(inputs)
    if dimensions:
        hf_outputs = matryoshka_fy(hf_outputs, dimensions)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
        tol=1e-2,
    )


def correctness_test_embed_models(
    hf_runner,
    aphrodite_runner,
    model_info: EmbedModelInfo,
    example_prompts,
    aphrodite_extra_kwargs=None,
    hf_model_callback=None,
):
    pytest.skip("Debug only, ci prefers to use mteb test.")

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and aphrodite_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    aphrodite_extra_kwargs = aphrodite_extra_kwargs or {}
    aphrodite_extra_kwargs["dtype"] = model_info.dtype

    if model_info.hf_overrides is not None:
        aphrodite_extra_kwargs["hf_overrides"] = model_info.hf_overrides

    with aphrodite_runner(
        model_info.name, runner="pooling", max_model_len=None, **aphrodite_extra_kwargs
    ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.embed(example_prompts)

    with hf_runner(
        model_info.name,
        dtype=model_info.hf_dtype,
        is_sentence_transformer=True,
    ) as hf_model:
        if hf_model_callback is not None:
            hf_model_callback(hf_model)

        run_embedding_correctness_test(hf_model, example_prompts, aphrodite_outputs)


async def run_client_embeddings(
    client: openai.AsyncOpenAI,
    model_name: str,
    queries: list[str],
    instruction: str = "",
) -> list[list[float]]:
    outputs = await client.embeddings.create(
        model=model_name,
        input=[instruction + q for q in queries],
    )
    return [data.embedding for data in outputs.data]
