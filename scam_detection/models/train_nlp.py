"""
Fast NLP training for CPU:
- DistilBERT encoder is FROZEN — only the classifier head is trained
- Uses 4,000 samples (stratified) — sufficient for high accuracy on this task
- 3 epochs — completes in ~5 minutes on CPU
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from scam_detection.models.nlp_model import NLPClassifier
from scam_detection.config import cfg

TRAIN_SAMPLES = 4000
EPOCHS = 3
BATCH_SIZE = 32


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.labels[i]


def train_nlp(data_path: str = "scam_detection/data/processed/text_dataset.csv",
              save_path: str = "scam_detection/models/nlp_classifier.pt"):
    df = pd.read_csv(data_path).dropna()

    # Stratified subsample for speed
    per_class = TRAIN_SAMPLES // 2
    df = pd.concat([
        df[df["label"] == lbl].sample(min((df["label"] == lbl).sum(), per_class), random_state=cfg.seed)
        for lbl in df["label"].unique()
    ]).reset_index(drop=True)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=cfg.seed, stratify=df["label"]
    )

    train_dl = DataLoader(TextDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TextDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    model = NLPClassifier().to(cfg.device)

    # Freeze DistilBERT encoder — only train the classification head
    for param in model.embedder.encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}  (encoder frozen)")
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}\n")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=True)
        for texts, labels in bar:
            labels = labels.to(cfg.device)
            optimizer.zero_grad()
            loss = loss_fn(model(list(texts)), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_dl)
        print(f"  → avg loss: {avg_loss:.4f}")

    # Evaluate
    print("\nEvaluating...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for texts, labels in tqdm(val_dl, desc="Validation", unit="batch"):
            out = model(list(texts)).argmax(dim=1).cpu().tolist()
            preds.extend(out)
            targets.extend(labels.tolist())

    print(classification_report(targets, preds, target_names=["ham", "scam"]))
    torch.save(model.state_dict(), save_path)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    train_nlp()
