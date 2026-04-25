"""Fusion model: concatenates NLP [768] + GNN [hidden_dim] embeddings → risk score."""
import torch
import torch.nn as nn
from scam_detection.config import cfg


class FusionModel(nn.Module):
    """
    Late-fusion MLP.
    Input:  nlp_emb [B, 768] + gnn_emb [B, hidden_dim]
    Output: logits  [B, 2]
    """

    def __init__(self, nlp_dim: int = 768, gnn_dim: int = None):
        super().__init__()
        gnn_dim = gnn_dim or cfg.gnn_hidden_dim
        in_dim = nlp_dim + gnn_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.fusion_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.num_classes),
        )

    def forward(self, nlp_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([nlp_emb, gnn_emb], dim=-1)  # [B, nlp_dim + gnn_dim]
        return self.net(x)                           # [B, 2]
