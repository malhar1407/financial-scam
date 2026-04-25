"""Graph Attention Network for transaction-level fraud detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from scam_detection.config import cfg


class GNNModel(nn.Module):
    """
    GAT encoder producing node embeddings.
    For node-level tasks (Elliptic): use node embeddings directly.
    For graph-level tasks: pool with global_mean_pool.
    """

    def __init__(self, in_dim: int = None, hidden_dim: int = None, num_layers: int = None):
        super().__init__()
        in_dim     = in_dim     or cfg.node_in_dim
        hidden_dim = hidden_dim or cfg.gnn_hidden_dim
        num_layers = num_layers or cfg.gnn_num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=4, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        self.dropout = nn.Dropout(cfg.gnn_dropout)
        self.embed_dim = hidden_dim

        # Node-level classifier head (used during standalone GNN training)
        self.classifier = nn.Linear(hidden_dim, cfg.num_classes)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings [N, hidden_dim]."""
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x  # [N, hidden_dim]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        """
        Returns logits.
        - Node-level (batch=None): [N, 2]
        - Graph-level (batch provided): [B, 2]
        """
        emb = self.encode(x, edge_index)
        if batch is not None:
            emb = global_mean_pool(emb, batch)
        return self.classifier(emb)
