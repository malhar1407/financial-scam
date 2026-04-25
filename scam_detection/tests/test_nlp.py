"""Tests for NLP module: embedder, LLM intent, RAG."""
import torch
import pytest
from unittest.mock import patch
from scam_detection.models.nlp_model import NLPClassifier
from scam_detection.models.nlp_pipeline import NLPPipeline


@pytest.fixture(scope="module")
def pipeline():
    return NLPPipeline(device="cpu")


def test_embedder_shape(pipeline):
    result = pipeline.run("Your account will be blocked. Send OTP now.")
    assert result.embedding.shape == (768,)
    assert result.nlp_logits.shape == (2,)


def test_embedding_is_different_for_different_texts(pipeline):
    r1 = pipeline.run("Win a lottery prize now!")
    r2 = pipeline.run("Your transaction of Rs 500 was successful.")
    assert not torch.allclose(r1.embedding, r2.embedding)


def test_llm_fallback_on_unavailable():
    """LLM intent extractor must not raise even if Ollama is offline."""
    from scam_detection.models.llm_intent import extract_intent
    with patch("requests.post", side_effect=ConnectionError("offline")):
        result = extract_intent("Test message")
    assert "intent" in result
    assert "risk_score" in result


def test_rag_returns_results(pipeline):
    result = pipeline.run("Your KYC is expired. Update now or account blocked.")
    assert isinstance(result.rag_matches, list)
    assert len(result.rag_matches) > 0
    assert "label" in result.rag_matches[0]
