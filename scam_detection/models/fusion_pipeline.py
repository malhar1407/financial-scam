"""End-to-end inference: text + graph → fused risk score."""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.data import Data
from scam_detection.models.nlp_pipeline import NLPPipeline, NLPResult
from scam_detection.models.gnn_pipeline import GNNPipeline, GNNResult
from scam_detection.models.fusion_model import FusionModel
from scam_detection.config import cfg


@dataclass
class FusionResult:
    risk_score: float          # 0.0 – 1.0
    label: str                 # "SCAM" | "SAFE"
    nlp: NLPResult
    gnn: GNNResult


class FusionPipeline:
    def __init__(
        self,
        nlp_weights: str = None,
        gnn_weights: str = None,
        fusion_weights: str = None,
        gnn_in_dim: int = None,
    ):
        self.nlp_pipe = NLPPipeline(device=cfg.device)
        if nlp_weights:
            self.nlp_pipe.load_weights(nlp_weights)

        self.gnn_pipe = GNNPipeline(weights_path=gnn_weights, in_dim=gnn_in_dim)

        self.fusion = FusionModel().to(cfg.device)
        if fusion_weights:
            self.fusion.load_state_dict(
                torch.load(fusion_weights, map_location=cfg.device)
            )
        self.fusion.eval()

    def run(self, text: str, graph: Data, target_node: int = 0) -> FusionResult:
        nlp_result = self.nlp_pipe.run(text)
        gnn_result = self.gnn_pipe.run(graph, target_node)

        nlp_emb = nlp_result.embedding.unsqueeze(0).to(cfg.device)   # [1, 768]
        gnn_emb = gnn_result.embedding.unsqueeze(0).to(cfg.device)   # [1, hidden_dim]

        with torch.no_grad():
            logits = self.fusion(nlp_emb, gnn_emb)                   # [1, 2]
            prob = F.softmax(logits, dim=-1)[0, 1].item()
            
            # Debug: individual model predictions
            nlp_prob = F.softmax(nlp_result.nlp_logits, dim=-1)[1].item()
            gnn_prob = F.softmax(gnn_result.gnn_logits, dim=-1)[1].item()
            
            # Fallback: if graph is trivial (single node, mostly zeros), use NLP only
            is_trivial_graph = (graph.x.shape[0] == 1 and graph.edge_index.shape[1] == 0)
            if is_trivial_graph:
                prob = nlp_prob
                print(f"[DEBUG] Trivial graph detected — using NLP-only: {prob:.3f}")
            else:
                print(f"[DEBUG] NLP: {nlp_prob:.3f}  |  GNN: {gnn_prob:.3f}  |  Fusion: {prob:.3f}")

        return FusionResult(
            risk_score=round(prob, 4),
            label="SCAM" if prob >= cfg.threshold else "SAFE",
            nlp=nlp_result,
            gnn=gnn_result,
        )
