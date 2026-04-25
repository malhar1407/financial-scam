"""Summarize GAT attention weights to identify suspicious neighbor nodes."""
import torch
import numpy as np
from torch_geometric.data import Data


def top_attended_neighbors(
    edge_index: torch.Tensor,
    attention_weights: list[torch.Tensor],
    target_node: int,
    k: int = 5,
) -> list[dict]:
    """
    For a target node, find the k neighbors with highest mean attention
    across all GAT layers.

    attention_weights: list of tensors [E, heads] captured from GATConv hooks.
    Returns list of {neighbor_node, mean_attention, layer_attentions}.
    """
    edge_index = edge_index.cpu()
    # Edges pointing TO target node
    dst_mask = edge_index[1] == target_node
    src_nodes = edge_index[0][dst_mask].numpy()

    if len(src_nodes) == 0:
        return []

    edge_positions = dst_mask.nonzero(as_tuple=True)[0].numpy()

    results = {}
    for node, pos in zip(src_nodes, edge_positions):
        layer_attn = []
        for layer_attn_tensor in attention_weights:
            if pos < len(layer_attn_tensor):
                # Mean over heads
                val = layer_attn_tensor[pos].mean().item()
                layer_attn.append(round(val, 4))
        mean_attn = float(np.mean(layer_attn)) if layer_attn else 0.0
        results[int(node)] = {"neighbor_node": int(node),
                               "mean_attention": round(mean_attn, 4),
                               "layer_attentions": layer_attn}

    sorted_neighbors = sorted(results.values(),
                               key=lambda x: x["mean_attention"], reverse=True)
    return sorted_neighbors[:k]
