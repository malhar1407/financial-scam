"""Tests for fusion pipeline and FastAPI endpoints."""
import torch
import pytest
from torch_geometric.data import Data
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from scam_detection.models.fusion_model import FusionModel
from scam_detection.models.fusion_pipeline import FusionPipeline
from scam_detection.config import cfg


def _make_graph(n_nodes=3, n_features=4):
    x = torch.randn(n_nodes, n_features)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# ── Fusion model ──────────────────────────────────────────────────────────

def test_fusion_model_output_shape():
    model = FusionModel()
    nlp = torch.randn(2, 768)
    gnn = torch.randn(2, cfg.gnn_hidden_dim)
    out = model(nlp, gnn)
    assert out.shape == (2, cfg.num_classes)


def test_fusion_pipeline_label():
    pipe = FusionPipeline(gnn_in_dim=4)
    graph = _make_graph()
    result = pipe.run("Win a lottery prize now!", graph)
    assert result.label in ("SCAM", "SAFE")
    assert 0.0 <= result.risk_score <= 1.0


# ── API ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    # Patch RAG store build to avoid ChromaDB I/O in CI
    with patch("scam_detection.api.app.build_rag_store"):
        from scam_detection.api.app import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_basic(client):
    payload = {
        "message": "Your SBI account will be blocked. Update KYC now.",
        "nodes": [{"node_id": 0, "features": [49999.0, 2.0, 5.0, 1.0]}],
        "edges": [],
        "target_node": 0,
        "explain": False,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "risk_score" in data
    assert data["label"] in ("SCAM", "SAFE")
    assert "nlp_intent" in data


def test_predict_with_edges(client):
    payload = {
        "message": "Send OTP to claim your reward.",
        "nodes": [
            {"node_id": 0, "features": [1000.0, 10.0, 3.0, 0.0]},
            {"node_id": 1, "features": [500.0, 11.0, 3.0, 1.0]},
        ],
        "edges": [{"src": 0, "dst": 1}],
        "target_node": 0,
        "explain": False,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
