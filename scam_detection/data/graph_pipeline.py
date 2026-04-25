"""Build transaction graph from Elliptic or PaySim dataset."""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def load_elliptic(
    features_path: str,
    edges_path: str,
    classes_path: str,
) -> Data:
    """
    Elliptic dataset files:
      elliptic_txs_features.csv  — node features (no header, col0=txId)
      elliptic_txs_edgelist.csv  — edges (txId1, txId2)
      elliptic_txs_classes.csv   — labels (txId, class: 1=illicit,2=licit,unknown)
    """
    feat_df = pd.read_csv(features_path, header=None)
    feat_df.columns = ["txId"] + [f"f{i}" for i in range(feat_df.shape[1] - 1)]

    classes_df = pd.read_csv(classes_path)
    classes_df["label"] = classes_df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})

    edges_df = pd.read_csv(edges_path)

    # Map txId → integer index
    all_ids = feat_df["txId"].tolist()
    id2idx = {tid: i for i, tid in enumerate(all_ids)}

    # Node features (drop txId col, normalize)
    x = feat_df.drop(columns=["txId"]).values.astype(np.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Edge index
    src = edges_df.iloc[:, 0].map(id2idx).dropna().astype(int)
    dst = edges_df.iloc[:, 1].map(id2idx).dropna().astype(int)
    edge_index = torch.tensor([src.tolist(), dst.tolist()], dtype=torch.long)

    # Labels (-1 for unknown)
    label_map = classes_df.set_index("txId")["label"]
    y_raw = feat_df["txId"].map(label_map).fillna(-1).astype(int).tolist()
    y = torch.tensor(y_raw, dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        y=y,
    )
    # Mask: only labeled nodes for training
    data.train_mask = y != -1
    return data


def load_paysim(csv_path: str) -> Data:
    """
    PaySim CSV columns: step, type, amount, nameOrig, oldbalanceOrg,
    newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud
    """
    df = pd.read_csv(csv_path)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

    nodes = pd.unique(df[["nameOrig", "nameDest"]].values.ravel())
    id2idx = {n: i for i, n in enumerate(nodes)}

    # Node features: aggregate per account
    def agg_features(name):
        sent = df[df["nameOrig"] == name]["amount"].sum()
        recv = df[df["nameDest"] == name]["amount"].sum()
        n_tx = len(df[(df["nameOrig"] == name) | (df["nameDest"] == name)])
        is_new = int(name.startswith("C") and n_tx < 3)
        return [sent, recv, n_tx, is_new]

    x = np.array([agg_features(n) for n in nodes], dtype=np.float32)
    x = StandardScaler().fit_transform(x)

    src = df["nameOrig"].map(id2idx).tolist()
    dst = df["nameDest"].map(id2idx).tolist()
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Node-level fraud: a node is fraud if it appears in any fraud tx
    fraud_orig = set(df[df["isFraud"] == 1]["nameOrig"])
    fraud_dest = set(df[df["isFraud"] == 1]["nameDest"])
    y = torch.tensor(
        [1 if n in fraud_orig or n in fraud_dest else 0 for n in nodes],
        dtype=torch.long,
    )

    return Data(x=torch.tensor(x), edge_index=edge_index, y=y)
