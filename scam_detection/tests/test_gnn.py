"""Tests for GNN module."""
import torch
import pytest
from torch_geometric.data import Data
from scam_detection.models.gnn_model import GNNModel
from scam_detection.models.gnn_pipeline import GNNPipeline
from scam_detection.config import cfg


def _make_graph(n_nodes=5, n_features=4):
    x = torch.randn(n_nodes, n_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_gnn_encode_shape():
    graph = _make_graph()
    model = GNNModel(in_dim=4)
    emb = model.encode(graph.x, graph.edge_index)
    assert emb.shape == (5, cfg.gnn_hidden_dim)


def test_gnn_forward_node_level():
    graph = _make_graph()
    model = GNNModel(in_dim=4)
    logits = model(graph.x, graph.edge_index)
    assert logits.shape == (5, cfg.num_classes)


def test_gnn_pipeline_result():
    graph = _make_graph()
    pipe = GNNPipeline(in_dim=4)
    result = pipe.run(graph, target_node=0)
    assert result.embedding.shape == (cfg.gnn_hidden_dim,)
    assert result.gnn_logits.shape == (cfg.num_classes,)
    assert isinstance(result.attention_weights, list)
