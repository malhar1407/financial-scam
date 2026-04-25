"""SHAP-based explainability for the FusionModel."""
import torch
import numpy as np
import shap
from scam_detection.models.fusion_model import FusionModel
from scam_detection.config import cfg


def _model_fn(fusion: FusionModel):
    """Wrap fusion model for SHAP: numpy in → numpy out (scam probability)."""
    def predict(x: np.ndarray) -> np.ndarray:
        t = torch.tensor(x, dtype=torch.float).to(cfg.device)
        nlp_dim = 768
        nlp_emb = t[:, :nlp_dim]
        gnn_emb = t[:, nlp_dim:]
        with torch.no_grad():
            logits = fusion(nlp_emb, gnn_emb)
            probs = torch.softmax(logits, dim=-1)[:, 1]
        return probs.cpu().numpy()
    return predict


def explain_fusion(
    fusion: FusionModel,
    nlp_emb: torch.Tensor,   # [768]
    gnn_emb: torch.Tensor,   # [hidden_dim]
    background_nlp: torch.Tensor,  # [N, 768]  background samples
    background_gnn: torch.Tensor,  # [N, hidden_dim]
) -> dict:
    """
    Returns SHAP values split into NLP and GNN contribution scores.
    Uses KernelExplainer (model-agnostic, works on any black-box).
    """
    bg = torch.cat([background_nlp, background_gnn], dim=-1).cpu().numpy()
    x  = torch.cat([nlp_emb.unsqueeze(0), gnn_emb.unsqueeze(0)], dim=-1).cpu().numpy()

    explainer = shap.KernelExplainer(_model_fn(fusion), bg)
    shap_vals = explainer.shap_values(x, nsamples=100, silent=True)  # [1, D]

    nlp_dim = nlp_emb.shape[0]
    nlp_shap = shap_vals[0][:nlp_dim]   # contribution from NLP features
    gnn_shap = shap_vals[0][nlp_dim:]   # contribution from GNN features

    return {
        "nlp_total_contribution": float(np.abs(nlp_shap).sum()),
        "gnn_total_contribution": float(np.abs(gnn_shap).sum()),
        "top_nlp_dims": _top_dims(nlp_shap, k=5),
        "top_gnn_dims": _top_dims(gnn_shap, k=5),
        "base_value": float(explainer.expected_value),
    }


def _top_dims(shap_arr: np.ndarray, k: int) -> list[dict]:
    idx = np.argsort(np.abs(shap_arr))[::-1][:k]
    return [{"dim": int(i), "shap": float(shap_arr[i])} for i in idx]
