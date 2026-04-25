"""GNN inference: embed a transaction node given its local subgraph."""
import torch
from dataclasses import dataclass
from torch_geometric.data import Data
from scam_detection.models.gnn_model import GNNModel
from scam_detection.config import cfg


@dataclass
class GNNResult:
    embedding: torch.Tensor    # [hidden_dim] — fed into fusion model
    gnn_logits: torch.Tensor   # [2] — standalone GNN fraud probability
    attention_weights: list    # per-layer attention (for explainability)


class GNNPipeline:
    def __init__(self, weights_path: str = None, in_dim: int = None):
        self.device = cfg.device
        self.model = GNNModel(in_dim=in_dim or cfg.node_in_dim).to(self.device)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def run(self, data: Data, target_node_idx: int = 0) -> GNNResult:
        """
        data: torch_geometric Data with x [N, F] and edge_index [2, E]
        target_node_idx: which node to extract embedding for (default: 0)
        """
        data = data.to(self.device)
        attention_weights = []

        # Hook to capture attention coefficients from each GATConv layer
        def _hook(module, inp, out):
            if isinstance(out, tuple):
                attention_weights.append(out[1].detach().cpu())

        handles = [conv.register_forward_hook(_hook) for conv in self.model.convs]

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)   # [N, 2]
            emb = self.model.encode(data.x, data.edge_index)[target_node_idx]  # [hidden_dim]

        for h in handles:
            h.remove()

        return GNNResult(
            embedding=emb,
            gnn_logits=logits[target_node_idx],
            attention_weights=attention_weights,
        )
