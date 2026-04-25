"""
Train the FusionModel on paired (NLP embedding, GNN embedding, label) samples.

Since no real paired dataset exists, we simulate pairs:
  - scam label  → NLP embedding from scam text  + GNN embedding from fraud node
  - benign label → NLP embedding from ham text   + GNN embedding from clean node
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from scam_detection.models.nlp_model import NLPClassifier
from scam_detection.models.gnn_model import GNNModel
from scam_detection.data.graph_pipeline import load_elliptic
from scam_detection.models.fusion_model import FusionModel
from scam_detection.config import cfg

TEXT_CSV   = "scam_detection/data/processed/text_dataset.csv"
ELLIPTIC_F = "scam_detection/data/raw/elliptic_txs_features.csv"
ELLIPTIC_E = "scam_detection/data/raw/elliptic_txs_edgelist.csv"
ELLIPTIC_C = "scam_detection/data/raw/elliptic_txs_classes.csv"
NLP_WEIGHTS = "scam_detection/models/nlp_classifier.pt"
GNN_WEIGHTS = "scam_detection/models/gnn_model.pt"
SAVE_PATH   = "scam_detection/models/fusion_model.pt"


class FusionDataset(Dataset):
    def __init__(self, nlp_embs, gnn_embs, labels):
        self.nlp = torch.tensor(nlp_embs, dtype=torch.float)
        self.gnn = torch.tensor(gnn_embs, dtype=torch.float)
        self.y   = torch.tensor(labels,   dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.nlp[i], self.gnn[i], self.y[i]


def _build_nlp_embeddings(texts, labels, nlp_model):
    """Batch-encode texts → numpy array [N, 768]."""
    nlp_model.eval()
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), cfg.batch_size), desc="NLP embeddings"):
            batch = texts[i: i + cfg.batch_size]
            embs.append(nlp_model.embed(batch).cpu().numpy())
    return np.vstack(embs), np.array(labels)


def _build_gnn_embeddings(graph_data, gnn_model):
    """Encode all labeled nodes → numpy array [N, hidden_dim]."""
    gnn_model.eval()
    with torch.no_grad():
        embs = gnn_model.encode(graph_data.x, graph_data.edge_index).cpu().numpy()
    mask = graph_data.train_mask.cpu().numpy()
    labels = graph_data.y.cpu().numpy()
    return embs[mask], labels[mask]


def _pair_embeddings(nlp_embs, nlp_labels, gnn_embs, gnn_labels):
    """
    Pair NLP and GNN embeddings by label.
    Truncate to the smaller class count to keep balance.
    """
    paired_nlp, paired_gnn, paired_y = [], [], []
    for label in [0, 1]:
        n_idx = np.where(nlp_labels == label)[0]
        g_idx = np.where(gnn_labels == label)[0]
        n = min(len(n_idx), len(g_idx))
        np.random.seed(cfg.seed)
        n_idx = np.random.choice(n_idx, n, replace=False)
        g_idx = np.random.choice(g_idx, n, replace=False)
        paired_nlp.append(nlp_embs[n_idx])
        paired_gnn.append(gnn_embs[g_idx])
        paired_y.extend([label] * n)
    return (
        np.vstack(paired_nlp),
        np.vstack(paired_gnn),
        np.array(paired_y),
    )


def train_fusion():
    device = cfg.device

    # ── Load pre-trained encoders ──────────────────────────────────────────
    nlp_model = NLPClassifier().to(device)
    nlp_model.load_state_dict(torch.load(NLP_WEIGHTS, map_location=device))

    graph_data = load_elliptic(ELLIPTIC_F, ELLIPTIC_E, ELLIPTIC_C).to(device)
    gnn_model = GNNModel(in_dim=graph_data.x.shape[1]).to(device)
    gnn_model.load_state_dict(torch.load(GNN_WEIGHTS, map_location=device))

    # ── Build embeddings ───────────────────────────────────────────────────
    df = pd.read_csv(TEXT_CSV).dropna()
    nlp_embs, nlp_labels = _build_nlp_embeddings(
        df["text"].tolist(), df["label"].tolist(), nlp_model
    )
    gnn_embs, gnn_labels = _build_gnn_embeddings(graph_data, gnn_model)

    nlp_embs, gnn_embs, y = _pair_embeddings(nlp_embs, nlp_labels, gnn_embs, gnn_labels)

    # ── Train / val split ─────────────────────────────────────────────────
    X_nlp_tr, X_nlp_val, X_gnn_tr, X_gnn_val, y_tr, y_val = train_test_split(
        nlp_embs, gnn_embs, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )

    train_dl = DataLoader(FusionDataset(X_nlp_tr, X_gnn_tr, y_tr),
                          batch_size=cfg.batch_size, shuffle=True)
    val_dl   = DataLoader(FusionDataset(X_nlp_val, X_gnn_val, y_val),
                          batch_size=cfg.batch_size)

    # ── Train fusion MLP ──────────────────────────────────────────────────
    fusion = FusionModel(nlp_dim=nlp_embs.shape[1], gnn_dim=gnn_embs.shape[1]).to(device)
    optimizer = Adam(fusion.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(cfg.epochs), desc="Fusion Training"):
        fusion.train()
        total_loss = 0
        for nlp_b, gnn_b, y_b in tqdm(train_dl, desc=f"  Epoch {epoch+1}", leave=False):
            nlp_b, gnn_b, y_b = nlp_b.to(device), gnn_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = loss_fn(fusion(nlp_b, gnn_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{cfg.epochs}  loss={total_loss/len(train_dl):.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    fusion.eval()
    preds, targets = [], []
    with torch.no_grad():
        for nlp_b, gnn_b, y_b in val_dl:
            out = fusion(nlp_b.to(device), gnn_b.to(device)).argmax(dim=1).cpu().tolist()
            preds.extend(out)
            targets.extend(y_b.tolist())

    print(classification_report(targets, preds, target_names=["benign", "scam"]))
    torch.save(fusion.state_dict(), SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    train_fusion()
