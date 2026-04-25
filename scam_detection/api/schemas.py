"""API request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class TransactionNode(BaseModel):
    """A single node's features in the transaction graph."""
    node_id: int = 0
    features: List[float] = Field(..., description="Node feature vector")


class TransactionEdge(BaseModel):
    src: int
    dst: int


class PredictRequest(BaseModel):
    message: str = Field(..., description="SMS/email/payment note text")
    nodes: List[TransactionNode] = Field(..., description="Graph nodes with features")
    edges: List[TransactionEdge] = Field(..., description="Graph edges")
    target_node: int = Field(0, description="Index of the node to classify")
    explain: bool = Field(False, description="Include SHAP + attention explanation")


class ShapSummary(BaseModel):
    nlp_total_contribution: float
    gnn_total_contribution: float
    top_nlp_dims: List[dict]
    top_gnn_dims: List[dict]


class ExplanationResponse(BaseModel):
    shap_summary: Optional[ShapSummary]
    top_neighbors: List[dict]
    report: str


class PredictResponse(BaseModel):
    risk_score: float
    label: str                          # "SCAM" | "SAFE"
    nlp_intent: dict
    rag_matches: List[dict]
    explanation: Optional[ExplanationResponse] = None
