"""DistilBERT-based text embedder + scam classifier."""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from scam_detection.config import cfg


class TextEmbedder(nn.Module):
    """Produces a [CLS] embedding from DistilBERT."""

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model)
        self.encoder = AutoModel.from_pretrained(cfg.embed_model)
        self.embed_dim = self.encoder.config.hidden_size  # 768

    def forward(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
            return_tensors="pt",
        ).to(next(self.encoder.parameters()).device)
        out = self.encoder(**enc)
        return out.last_hidden_state[:, 0, :]  # [B, 768]


class NLPClassifier(nn.Module):
    """Thin classifier head on top of TextEmbedder (used standalone)."""

    def __init__(self):
        super().__init__()
        self.embedder = TextEmbedder()
        self.head = nn.Sequential(
            nn.Linear(self.embedder.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, cfg.num_classes),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        emb = self.embedder(texts)
        return self.head(emb)  # [B, 2] logits

    def embed(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.embedder(texts)  # [B, 768]
