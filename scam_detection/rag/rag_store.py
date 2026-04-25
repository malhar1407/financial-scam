"""RAG: vector store of known scam patterns using ChromaDB + sentence-transformers."""
import chromadb
from sentence_transformers import SentenceTransformer
from scam_detection.config import cfg

_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Seed patterns — extend with RBI/CERT-In advisories
_SEED_PATTERNS = [
    {"id": "p1", "text": "Your KYC is expired. Update immediately or your account will be blocked.", "label": "kyc_scam"},
    {"id": "p2", "text": "Congratulations! You have won a lottery of Rs 50 lakh. Send OTP to claim.", "label": "lottery_scam"},
    {"id": "p3", "text": "Dear customer, your SBI account is suspended. Click link to verify.", "label": "phishing"},
    {"id": "p4", "text": "URGENT: Unauthorized login detected. Call 9XXXXXXXXX immediately.", "label": "impersonation"},
    {"id": "p5", "text": "Send Rs 999 to activate your cashback reward before midnight.", "label": "urgency_scam"},
    {"id": "p6", "text": "Your UPI PIN has been compromised. Reset via this link now.", "label": "phishing"},
    {"id": "p7", "text": "Income Tax refund of Rs 15,420 approved. Submit bank details.", "label": "govt_impersonation"},
    {"id": "p8", "text": "Work from home and earn Rs 5000/day. Register with Rs 500 fee.", "label": "job_scam"},
]


def _get_collection():
    client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
    return client.get_or_create_collection("scam_patterns")


def build_rag_store():
    """Seed the vector store with known scam patterns."""
    col = _get_collection()
    existing = set(col.get()["ids"])
    new = [p for p in _SEED_PATTERNS if p["id"] not in existing]
    if not new:
        return
    col.add(
        ids=[p["id"] for p in new],
        embeddings=_EMBED_MODEL.encode([p["text"] for p in new]).tolist(),
        documents=[p["text"] for p in new],
        metadatas=[{"label": p["label"]} for p in new],
    )
    print(f"RAG store: added {len(new)} patterns.")


def retrieve_similar(text: str) -> list[dict]:
    """Return top-k similar known scam patterns for a given message."""
    col = _get_collection()
    emb = _EMBED_MODEL.encode([text]).tolist()
    results = col.query(query_embeddings=emb, n_results=cfg.rag_top_k)
    return [
        {"text": doc, "label": meta["label"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


if __name__ == "__main__":
    build_rag_store()
    hits = retrieve_similar("Your HDFC account will be blocked. Update KYC now.")
    for h in hits:
        print(h)
