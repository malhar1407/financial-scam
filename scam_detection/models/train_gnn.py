"""Train GNNModel on the Elliptic transaction graph (node classification)."""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
from scam_detection.models.gnn_model import GNNModel
from scam_detection.data.graph_pipeline import load_elliptic
from scam_detection.config import cfg

FEATURES  = "scam_detection/data/raw/elliptic_txs_features.csv"
EDGES     = "scam_detection/data/raw/elliptic_txs_edgelist.csv"
CLASSES   = "scam_detection/data/raw/elliptic_txs_classes.csv"
SAVE_PATH = "scam_detection/models/gnn_model.pt"
EPOCHS    = 50
LR        = 5e-3


def focal_loss(logits, targets, weight, gamma=2.0):
    """Focal loss: down-weights easy examples, focuses on hard minority class."""
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def train_gnn():
    print("Loading Elliptic graph...")
    data = load_elliptic(FEATURES, EDGES, CLASSES).to(cfg.device)

    labeled_idx = data.train_mask.nonzero(as_tuple=True)[0]
    split = int(0.8 * len(labeled_idx))
    train_idx, val_idx = labeled_idx[:split], labeled_idx[split:]

    # Class weights
    train_labels = data.y[train_idx]
    n_licit   = (train_labels == 0).sum().item()
    n_illicit = (train_labels == 1).sum().item()
    weight = torch.tensor([1.0, n_licit / max(n_illicit, 1)], dtype=torch.float).to(cfg.device)

    print(f"Nodes: {data.x.shape[0]:,}  |  Train: {len(train_idx):,}  Val: {len(val_idx):,}")
    print(f"Illicit in train: {n_illicit}  |  Class weight: {weight[1]:.1f}x\n")

    model = GNNModel(in_dim=data.x.shape[1]).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_f1, best_state = 0.0, None
    bar = tqdm(range(EPOCHS), desc="Training GNN", unit="epoch")

    for epoch in bar:
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = focal_loss(logits[train_idx], data.y[train_idx], weight)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track best model by illicit F1 on val set
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x, data.edge_index)
                preds = val_logits[val_idx].argmax(dim=1).cpu()
                targets = data.y[val_idx].cpu()
            illicit_mask = targets == 1
            if illicit_mask.sum() > 0:
                tp = ((preds == 1) & illicit_mask).sum().item()
                fp = ((preds == 1) & ~illicit_mask).sum().item()
                fn = ((preds == 0) & illicit_mask).sum().item()
                prec = tp / max(tp + fp, 1)
                rec  = tp / max(tp + fn, 1)
                f1   = 2 * prec * rec / max(prec + rec, 1e-8)
                bar.set_postfix(loss=f"{loss.item():.4f}", illicit_f1=f"{f1:.3f}")
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final evaluation with best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds   = logits[val_idx].argmax(dim=1).cpu()
        targets = data.y[val_idx].cpu()

    print("\n" + classification_report(targets, preds,
                                       target_names=["licit", "illicit"],
                                       zero_division=0))
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    train_gnn()
