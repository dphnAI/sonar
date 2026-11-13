# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the LlamaNemotronVL model family:
  - nvidia/llama-nemotron-embed-vl-1b-v2  (LlamaNemotronVLForCausalLM / embed)
  - nvidia/llama-nemotron-rerank-vl-1b-v2
      (LlamaNemotronVLForSequenceClassification / rerank)

Both variants share a SigLIP vision encoder with a bidirectional LLaMA backbone.
"""

from collections.abc import Sequence
from io import BytesIO

import pybase64 as base64
import pytest
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor

from sonar.entrypoints.pooling.scoring.types import ScoreMultiModalParam
from sonar.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm
from sonar.platforms import current_platform
from sonar.renderers.chat_utils import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from sonar.transformers_utils.processors.nemotron_vl import (
    SIGLIP_MEAN,
    SIGLIP_STD,
    LlamaNemotronVLEmbedImageProcessor,
)

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, SonarRunner
from ....utils import ROCM_ENGINE_KWARGS
from ...utils import check_embeddings_close

# Prefixes used by the model API
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Text prompts for text-only embedding
HF_TEXT_PROMPTS = [
    # T -> X (text embedding queries)
    f"{QUERY_PREFIX}The label of the object is stop sign",
    f"{QUERY_PREFIX}cherry blossom",
]

# Image prompts using the model's expected format
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        # I -> X (image embedding as passage/document)
        "stop_sign": f"{PASSAGE_PREFIX}<image>",
        "cherry_blossom": f"{PASSAGE_PREFIX}<image>",
    }
)

MODELS = ["nvidia/llama-nemotron-embed-vl-1b-v2"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    ("async_scheduling", "long_prefill_token_threshold"),
    [(True, 0), (False, 128)],
    ids=["async-encoder-cache-pressure", "sync-prefill-threshold"],
)
def test_image_embeddings_without_chunked_prefill(
    sonar_runner, model, monkeypatch, async_scheduling, long_prefill_token_threshold
):
    """Encoder pressure and prefill caps must preserve whole bidirectional prompts."""
    # Queue both requests before scheduling, independent of arrival timing.
    monkeypatch.setenv("SONAR_ENABLE_V1_MULTIPROCESSING", "0")
    images = [Image.new("RGB", (512, 512), color=color) for color in ("red", "blue")]
    prompts = [f"{PASSAGE_PREFIX}<image>"] * len(images)
    # ROCM_ENGINE_KWARGS also sets max_num_seqs, which this test needs at 2.
    engine_kwargs = {**ROCM_ENGINE_KWARGS, "max_num_seqs": 2}
    # The 384-token encoder cache fits one 256-token image, but not both.
    with sonar_runner(
        model,
        runner="pooling",
        dtype="half",
        max_model_len=384,
        max_num_batched_tokens=384,
        async_scheduling=async_scheduling,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=False,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_dynamic_patch": 1, "use_thumbnail": False},
        **engine_kwargs,
    ) as sonar_model:
        batched = sonar_model.embed(prompts, images=images)
        individual = [sonar_model.embed([prompt], images=[image])[0] for prompt, image in zip(prompts, images)]
    check_embeddings_close(
        embeddings_0_lst=individual,
        embeddings_1_lst=batched,
        name_0="individual",
        name_1="batch",
    )


@pytest.mark.usefixtures("default_sonar_config")
def test_device_normalization_matches_cpu_transform() -> None:
    device = current_platform.device_type
    image = Image.new("RGB", (48, 32), color=(12, 128, 240))
    processor_kwargs = dict(
        image_size=32,
        min_dynamic_patch=1,
        max_dynamic_patch=1,
        dynamic_image_size=True,
        use_thumbnail=False,
    )

    cpu_processor = LlamaNemotronVLEmbedImageProcessor(**processor_kwargs)
    device_processor = LlamaNemotronVLEmbedImageProcessor(**processor_kwargs, do_rescale=False, do_normalize=False)

    expected = cpu_processor._images_to_pixel_values_lst([image])[0]
    raw_pixels = device_processor._images_to_pixel_values_lst([image])[0]
    input_norm = FusedMMInputNorm(
        image_mean=list(SIGLIP_MEAN),
        image_std=list(SIGLIP_STD),
        rescale_factor=1 / 255,
    ).to(device)
    actual = input_norm(raw_pixels.flatten(start_dim=1).to(device), expected.dtype).view_as(expected)

    assert raw_pixels.dtype == torch.uint8
    torch.testing.assert_close(actual, expected.to(device))


@pytest.mark.parametrize(("do_rescale", "do_normalize"), [(True, False), (False, True)])
def test_device_normalization_rejects_partial_cpu_processing(do_rescale: bool, do_normalize: bool) -> None:
    with pytest.raises(ValueError, match="both enabled or both disabled"):
        LlamaNemotronVLEmbedImageProcessor(
            image_size=32,
            min_dynamic_patch=1,
            max_dynamic_patch=1,
            dynamic_image_size=True,
            use_thumbnail=False,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
        )


def _run_test(
    hf_runner: type[HfRunner],
    sonar_runner: type[SonarRunner],
    input_cases: list[tuple[list[str], PromptImageInput]],
    model: str,
    *,
    dtype: str,
) -> None:
    """Compare HF and Sonar embeddings for all input cases.

    NOTE: Run Sonar first to avoid CUDA initialization issues with multiprocessing.
    """
    with sonar_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
        **ROCM_ENGINE_KWARGS,
    ) as sonar_model:
        sonar_outputs_per_case = [
            sonar_model.embed(input_texts, images=input_images) for input_texts, input_images in input_cases
        ]

    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs_per_case = []
        for input_texts, input_images in input_cases:
            hf_outputs = []
            for text, image in zip(input_texts, input_images):
                with torch.inference_mode():
                    if text.startswith(QUERY_PREFIX):
                        query_text = text[len(QUERY_PREFIX) :]
                        embedding = hf_model.model.encode_queries([query_text])
                    elif text.startswith(PASSAGE_PREFIX):
                        passage_text = text[len(PASSAGE_PREFIX) :]
                        if image is not None:
                            embedding = hf_model.model.encode_documents(
                                images=[image],
                                texts=[passage_text],
                            )
                        else:
                            embedding = hf_model.model.encode_documents(texts=[passage_text])
                    else:
                        raise ValueError(f"Text must start with {QUERY_PREFIX!r} or {PASSAGE_PREFIX!r}")

                    hf_outputs.append(embedding[0].tolist())
            hf_outputs_per_case.append(hf_outputs)

    for hf_outputs, sonar_outputs in zip(hf_outputs_per_case, sonar_outputs_per_case):
        check_embeddings_close(
            embeddings_0_lst=hf_outputs,
            embeddings_1_lst=sonar_outputs,
            name_0="hf",
            name_1="sonar",
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    hf_runner,
    sonar_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test text and image embedding."""
    text_images = [None] * len(HF_TEXT_PROMPTS)
    images = [asset.pil_image for asset in image_assets]
    input_cases = [
        (HF_TEXT_PROMPTS, text_images),
        (HF_IMAGE_PROMPTS, images),
    ]

    _run_test(
        hf_runner,
        sonar_runner,
        input_cases,
        model,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Reranker tests — nvidia/llama-nemotron-rerank-vl-1b-v2
# ---------------------------------------------------------------------------

RERANKER_MODELS = ["nvidia/llama-nemotron-rerank-vl-1b-v2"]

RERANKER_TEXT_QUERY = "How is AI improving the intelligence and capabilities of robots?"
RERANKER_TEXT_DOCS = [
    "AI enables robots to perceive, plan, and act autonomously.",
    ("A biological foundation model designed to analyze DNA, RNA, and protein sequences."),
]

RERANKER_IMAGE_QUERY = "photo of a red stop sign on a street"

RerankerDocument = tuple[str | None, Image.Image | None]
RerankerCase = tuple[str, Sequence[RerankerDocument]]


def _pil_to_data_uri(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _run_hf_reranker(
    hf_runner: type[HfRunner],
    model: str,
    dtype: str,
    input_cases: Sequence[RerankerCase],
) -> list[list[float]]:
    """Run all HF reranker cases in one model lifecycle."""
    with hf_runner(
        model,
        dtype=dtype,
        trust_remote_code=True,
        auto_cls=AutoModelForSequenceClassification,
    ) as hf_model:
        processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
            max_input_tiles=6,
            use_thumbnail=True,
            rerank_max_length=2048,
        )
        scores_per_case = []
        for query, docs in input_cases:
            examples = [
                {
                    "question": query,
                    "doc_text": doc_text if doc_text is not None else "",
                    "doc_image": doc_image if doc_image is not None else "",
                }
                for doc_text, doc_image in docs
            ]
            batch_dict = processor.process_queries_documents_crossencoder(examples)
            batch_dict = {
                k: v.to(hf_model.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()
            }
            with torch.inference_mode():
                logits = hf_model.model(**batch_dict, return_dict=True).logits
            # Sonar applies sigmoid activation to raw logits before returning scores.
            scores = torch.sigmoid(logits.squeeze(-1).float())
            scores_per_case.append(scores.detach().cpu().tolist())

    return scores_per_case


def _run_sonar_reranker(
    sonar_runner: type[SonarRunner],
    model: str,
    dtype: str,
    input_cases: Sequence[RerankerCase],
) -> list[list[float]]:
    """Run all Sonar reranker cases in one model lifecycle.

    The checkpoint declares its score template in its Sentence Transformers
    config, so no chat template is passed.
    """
    with sonar_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
        **ROCM_ENGINE_KWARGS,
    ) as sonar_model:
        scores_per_case = []
        for query, docs in input_cases:
            has_images = any(img is not None for _, img in docs)

            if not has_images:
                queries = [query] * len(docs)
                doc_texts = [doc_text for doc_text, _ in docs]
                outputs = sonar_model.score(queries, doc_texts)
            else:
                query_params = [
                    ScoreMultiModalParam(
                        content=[
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=query,
                            )
                        ]
                    )
                ] * len(docs)

                doc_params = []
                for doc_text, doc_image in docs:
                    content: list = []
                    if doc_image is not None:
                        content.append(
                            ChatCompletionContentPartImageParam(
                                type="image_url",
                                image_url={"url": _pil_to_data_uri(doc_image)},
                            )
                        )
                    if doc_text:
                        content.append(
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=doc_text,
                            )
                        )
                    doc_params.append(ScoreMultiModalParam(content=content))

                raw_outputs = sonar_model.llm.score(query_params, doc_params)
                outputs = [output.outputs.score for output in raw_outputs]

            scores_per_case.append(outputs)

    return scores_per_case


def _run_reranker_test(
    hf_runner: type[HfRunner],
    sonar_runner: type[SonarRunner],
    model: str,
    dtype: str,
    input_cases: Sequence[RerankerCase],
) -> None:
    """Compare HF and Sonar reranker scores for all input cases.

    NOTE: Run Sonar first to avoid CUDA initialization issues with multiprocessing.
    """
    sonar_scores_per_case = _run_sonar_reranker(sonar_runner, model, dtype, input_cases)
    hf_scores_per_case = _run_hf_reranker(hf_runner, model, dtype, input_cases)

    # ROCm has slightly higher variance because Sonar and HF use different
    # attention backends.
    rel_tol = 0.022 if current_platform.is_rocm() else 0.02
    for hf_scores, sonar_scores in zip(hf_scores_per_case, sonar_scores_per_case):
        assert len(hf_scores) == len(sonar_scores), (
            f"Output length mismatch: HF={len(hf_scores)}, Sonar={len(sonar_scores)}"
        )
        for i, (hf_score, sonar_score) in enumerate(zip(hf_scores, sonar_scores)):
            assert hf_score == pytest.approx(sonar_score, rel=rel_tol), (
                f"Score mismatch at index {i}: HF={hf_score:.4f}, Sonar={sonar_score:.4f}"
            )


@pytest.mark.parametrize("model", RERANKER_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_reranker(
    hf_runner,
    sonar_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test reranking with text and image documents."""
    text_docs: list[RerankerDocument] = [(text, None) for text in RERANKER_TEXT_DOCS]
    image_docs: list[RerankerDocument] = [(None, asset.pil_image) for asset in image_assets]
    input_cases: list[RerankerCase] = [
        (RERANKER_TEXT_QUERY, text_docs),
        (RERANKER_IMAGE_QUERY, image_docs),
    ]
    _run_reranker_test(hf_runner, sonar_runner, model, dtype, input_cases)
