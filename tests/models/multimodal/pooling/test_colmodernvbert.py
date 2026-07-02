# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColModernVBERT multimodal late-interaction model.

ColModernVBERT combines SigLIP vision encoder + ModernBERT text encoder
with a pixel shuffle connector and ColBERT-style 128-dim per-token
embeddings for visual document retrieval.
"""

import pytest
import torch

from aphrodite.entrypoints.pooling.scoring.utils import compute_maxsim_score

MODEL_NAME = "ModernVBERT/colmodernvbert-merged"
COLBERT_DIM = 128
DTYPE = "half"


# -----------------------------------------------------------------------
# Text-only tests
# -----------------------------------------------------------------------


def test_colmodernvbert_text_token_embed(aphrodite_runner):
    """Text query produces per-token embeddings with shape (seq_len, 128)."""
    with aphrodite_runner(
        MODEL_NAME,
        runner="pooling",
        dtype=DTYPE,
        enforce_eager=True,
    ) as aphrodite_model:
        outputs = aphrodite_model.token_embed(["What is machine learning?"])

        assert len(outputs) == 1
        emb = torch.tensor(outputs[0])
        assert emb.dim() == 2
        assert emb.shape[1] == COLBERT_DIM
        assert emb.shape[0] > 1


def test_colmodernvbert_text_relevance_ordering(aphrodite_runner):
    """Relevant documents score higher than irrelevant ones."""
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather in Paris is mild in spring.",
    ]

    with aphrodite_runner(
        MODEL_NAME,
        runner="pooling",
        dtype=DTYPE,
        enforce_eager=True,
    ) as aphrodite_model:
        scores = aphrodite_model.score(query, documents)

        assert len(scores) == 2
        assert scores[0] > scores[1], "ML doc should score higher than weather doc"


def test_colmodernvbert_text_late_interaction(aphrodite_runner):
    """MaxSim scoring via Aphrodite matches manual computation."""
    query = "What is the capital of France?"
    doc = "The capital of France is Paris."

    with aphrodite_runner(
        MODEL_NAME,
        runner="pooling",
        dtype=DTYPE,
        enforce_eager=True,
    ) as aphrodite_model:
        q_out = aphrodite_model.token_embed([query])
        d_out = aphrodite_model.token_embed([doc])

        q_emb = torch.tensor(q_out[0])
        d_emb = torch.tensor(d_out[0])
        manual_score = compute_maxsim_score(q_emb, d_emb).item()

        aphrodite_scores = aphrodite_model.score(query, doc)

        assert len(aphrodite_scores) == 1
        assert aphrodite_scores[0] == pytest.approx(manual_score, rel=0.01)


# -----------------------------------------------------------------------
# Image tests
# -----------------------------------------------------------------------


def test_colmodernvbert_image_token_embed(aphrodite_runner, image_assets):
    """Image input produces per-token embeddings including vision tokens."""
    with aphrodite_runner(
        MODEL_NAME,
        runner="pooling",
        dtype=DTYPE,
        enforce_eager=True,
    ) as aphrodite_model:
        image = image_assets[0].pil_image
        inputs = aphrodite_model.get_inputs(
            [""],
            images=[image],
        )
        req_outputs = aphrodite_model.llm.encode(
            inputs,
            pooling_task="token_embed",
        )
        outputs = [req_output.outputs.data for req_output in req_outputs]

        assert len(outputs) == 1
        emb = torch.tensor(outputs[0])
        assert emb.dim() == 2
        assert emb.shape[1] == COLBERT_DIM
        # Should have at least the image tokens (64 after pixel shuffle)
        assert emb.shape[0] >= 64
