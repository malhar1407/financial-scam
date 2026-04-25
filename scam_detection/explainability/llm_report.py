"""Generate a human-readable risk report using the LLM."""
import json
import requests
from scam_detection.config import cfg

_SYSTEM = """You are a financial fraud analyst. Given structured detection signals,
write a concise 3-sentence risk report for a compliance officer.
Be specific about what triggered the alert. Do not use bullet points."""


def generate_report(
    text: str,
    risk_score: float,
    label: str,
    llm_intent: dict,
    rag_matches: list[dict],
    shap_summary: dict,
    top_neighbors: list[dict],
) -> str:
    """Call Ollama LLM to produce a natural language explanation."""
    context = {
        "message": text,
        "risk_score": risk_score,
        "label": label,
        "detected_intent": llm_intent.get("intent"),
        "tactics": llm_intent.get("tactics", []),
        "llm_reason": llm_intent.get("reason"),
        "similar_known_scam": rag_matches[0]["label"] if rag_matches else "none",
        "rag_similarity_distance": rag_matches[0]["distance"] if rag_matches else None,
        "nlp_contribution": shap_summary.get("nlp_total_contribution"),
        "gnn_contribution": shap_summary.get("gnn_total_contribution"),
        "suspicious_neighbors": [n["neighbor_node"] for n in top_neighbors[:3]],
    }

    payload = {
        "model": cfg.llm_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Detection signals:\n{json.dumps(context, indent=2)}"},
        ],
    }
    try:
        resp = requests.post(
            f"{cfg.ollama_base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        # Fallback: structured summary without LLM
        return (
            f"[{label}] Risk score: {risk_score:.2f}. "
            f"Intent: {llm_intent.get('intent', 'unknown')}. "
            f"Tactics: {', '.join(llm_intent.get('tactics', []))}. "
            f"LLM unavailable: {e}"
        )
