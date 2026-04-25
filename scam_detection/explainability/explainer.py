"""Unified explainability: SHAP + attention + LLM report."""
import torch
from dataclasses import dataclass
from torch_geometric.data import Data
from scam_detection.models.fusion_model import FusionModel
from scam_detection.models.fusion_pipeline import FusionResult
from scam_detection.explainability.shap_explainer import explain_fusion
from scam_detection.explainability.attention_explainer import top_attended_neighbors
from scam_detection.explainability.llm_report import generate_report


@dataclass
class Explanation:
    shap_summary: dict          # NLP vs GNN contribution + top dims
    top_neighbors: list[dict]   # most attended transaction neighbors
    report: str                 # LLM-generated natural language summary


def explain(
    text: str,
    graph: Data,
    result: FusionResult,
    fusion_model: FusionModel,
    background_nlp: torch.Tensor,   # [N, 768]  — sample of training embeddings
    background_gnn: torch.Tensor,   # [N, hidden_dim]
    target_node: int = 0,
) -> Explanation:
    # 1. SHAP on fusion MLP
    shap_summary = explain_fusion(
        fusion_model,
        result.nlp.embedding,
        result.gnn.embedding,
        background_nlp,
        background_gnn,
    )

    # 2. GAT attention → suspicious neighbors
    neighbors = top_attended_neighbors(
        graph.edge_index,
        result.gnn.attention_weights,
        target_node=target_node,
    )

    # 3. LLM natural language report
    report = generate_report(
        text=text,
        risk_score=result.risk_score,
        label=result.label,
        llm_intent=result.nlp.llm_intent,
        rag_matches=result.nlp.rag_matches,
        shap_summary=shap_summary,
        top_neighbors=neighbors,
    )

    return Explanation(
        shap_summary=shap_summary,
        top_neighbors=neighbors,
        report=report,
    )
