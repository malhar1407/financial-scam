from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # NLP / LLM
    embed_model: str = "distilbert-base-uncased"
    llm_model: str = "mistral"          # Ollama model name
    ollama_base_url: str = "http://localhost:11434"
    max_seq_len: int = 128

    # RAG
    chroma_persist_dir: str = "scam_detection/rag/chroma_db"
    rag_top_k: int = 3

    # GNN
    node_in_dim: int = 4
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.3

    # Fusion
    fusion_hidden_dim: int = 128
    num_classes: int = 2

    # Training
    lr: float = 2e-5
    batch_size: int = 32
    epochs: int = 10
    seed: int = 42

    # Inference
    threshold: float = 0.5
    device: str = "cpu"

    node_feature_cols: List[str] = field(default_factory=lambda: [
        "amount", "hour", "day_of_week", "is_new_account"
    ])

cfg = Config()
