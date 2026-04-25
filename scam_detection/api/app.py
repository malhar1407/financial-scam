"""FastAPI real-time scam detection API."""
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from torch_geometric.data import Data

from scam_detection.api.schemas import PredictRequest, PredictResponse, ExplanationResponse
from scam_detection.models.fusion_pipeline import FusionPipeline
from scam_detection.explainability.explainer import explain
from scam_detection.rag.rag_store import build_rag_store
from scam_detection.config import cfg

# ── Model paths (set to None to use untrained weights for dev/testing) ────
NLP_WEIGHTS    = "scam_detection/models/nlp_classifier.pt"
GNN_WEIGHTS    = "scam_detection/models/gnn_model.pt"
FUSION_WEIGHTS = "scam_detection/models/fusion_model.pt"

_pipeline: FusionPipeline = None


def _load_weights(path: str):
    import os
    return path if os.path.exists(path) else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    build_rag_store()
    
    # Auto-detect GNN input dim from saved weights
    gnn_in_dim = cfg.node_in_dim
    gnn_path = _load_weights(GNN_WEIGHTS)
    if gnn_path:
        state = torch.load(gnn_path, map_location="cpu")
        # First conv layer weight shape: [out_features, in_features]
        gnn_in_dim = state["convs.0.lin.weight"].shape[1]
    
    _pipeline = FusionPipeline(
        nlp_weights=_load_weights(NLP_WEIGHTS),
        gnn_weights=gnn_path,
        fusion_weights=_load_weights(FUSION_WEIGHTS),
        gnn_in_dim=gnn_in_dim,
    )
    yield


app = FastAPI(
    title="Financial Scam Detection API",
    description="Multi-modal NLP + GNN scam detection with explainability",
    version="1.0.0",
    lifespan=lifespan,
)


def _build_graph(request: PredictRequest) -> Data:
    """Convert API request payload into a PyG Data object."""
    x = torch.tensor(
        [n.features for n in request.nodes], dtype=torch.float
    )
    if request.edges:
        edge_index = torch.tensor(
            [[e.src for e in request.edges], [e.dst for e in request.edges]],
            dtype=torch.long,
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Pad/truncate node features to match GNN's expected input dim
    expected = _pipeline.gnn_pipe.model.convs[0].lin.weight.shape[1]
    if x.shape[1] != expected:
        if x.shape[1] < expected:
            pad = torch.zeros(x.shape[0], expected - x.shape[1])
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :expected]

    return Data(x=x, edge_index=edge_index)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    graph = _build_graph(request)
    result = _pipeline.run(request.message, graph, request.target_node)

    explanation_resp = None
    if request.explain:
        try:
            # Background embeddings: use zero vectors as neutral baseline
            bg_nlp = torch.zeros(10, 768)
            bg_gnn = torch.zeros(10, cfg.gnn_hidden_dim)
            exp = explain(
                text=request.message,
                graph=graph,
                result=result,
                fusion_model=_pipeline.fusion,
                background_nlp=bg_nlp,
                background_gnn=bg_gnn,
                target_node=request.target_node,
            )
            explanation_resp = ExplanationResponse(
                shap_summary=exp.shap_summary,
                top_neighbors=exp.top_neighbors,
                report=exp.report,
            )
        except Exception as e:
            print(f"[WARNING] Explainability failed: {e}")
            explanation_resp = ExplanationResponse(
                shap_summary=None,
                top_neighbors=[],
                report=f"Explainability unavailable: {type(e).__name__}",
            )

    return PredictResponse(
        risk_score=result.risk_score,
        label=result.label,
        nlp_intent=result.nlp.llm_intent,
        rag_matches=result.nlp.rag_matches,
        explanation=explanation_resp,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}
