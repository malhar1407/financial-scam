"""LLM-based intent extractor using Ollama (Mistral/Llama)."""
import json
import requests
from scam_detection.config import cfg

_SYSTEM_PROMPT = """You are a financial scam detection expert.
Analyze the message and return ONLY valid JSON with these fields:
- intent: one of [phishing, impersonation, urgency_scam, lottery_scam, benign]
- tactics: list of manipulation tactics detected (e.g. urgency, authority, fear)
- entities: list of extracted entities (bank names, amounts, URLs, phone numbers)
- risk_score: float 0.0-1.0 (your confidence this is a scam)
- reason: one sentence explanation"""


def extract_intent(text: str) -> dict:
    """Call local Ollama LLM to extract structured scam signals."""
    payload = {
        "model": cfg.llm_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Message: {text}"},
        ],
    }
    try:
        resp = requests.post(
            f"{cfg.ollama_base_url}/api/chat",
            json=payload,
            timeout=30,  # Very long timeout for CPU inference
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Strip markdown code fences if present
        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(content)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        # Graceful fallback — use simple heuristics
        text_lower = text.lower()
        intent = "unknown"
        tactics = []
        if any(w in text_lower for w in ["urgent", "immediately", "now", "expire"]):
            tactics.append("urgency")
        if any(w in text_lower for w in ["account", "bank", "kyc", "verify"]):
            intent = "phishing"
            tactics.append("impersonation")
        if any(w in text_lower for w in ["won", "prize", "lottery", "claim"]):
            intent = "lottery_scam"
        
        return {
            "intent": intent,
            "tactics": tactics,
            "entities": [],
            "risk_score": 0.7 if tactics else 0.3,
            "reason": f"LLM unavailable (using heuristics): {type(e).__name__}",
        }
