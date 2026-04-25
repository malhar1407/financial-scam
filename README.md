# Financial Scam Detection — Multi-Modal NLP + GNN Framework

Early detection of financial scams and fraudulent messages using context-aware NLP, Graph Neural Networks, LLM-based intent analysis, and RAG-powered pattern memory.

---

## Architecture

```
Input: Message + Transaction Graph
        │
        ├──► DistilBERT Embedder  ──► [768-dim embedding]
        ├──► Mistral LLM (Ollama) ──► {intent, tactics, entities, risk_score}
        ├──► RAG (ChromaDB)       ──► top-k similar known scam patterns
        │
        ├──► GAT (4-head)         ──► [64-dim node embedding]
        │       └── attention weights captured for explainability
        │
        ├──► Fusion MLP (832 → 128 → 64 → 2)
        │       └── risk_score (0–1) + label (SCAM/SAFE)
        │
        └──► Explainability
                ├── SHAP: NLP vs GNN contribution
                ├── Attention: suspicious neighbor nodes
                └── LLM: 3-sentence compliance report
```

---

## Project Structure

```
scam_detection/
├── config.py                    # All hyperparameters and paths
├── requirements.txt
├── data/
│   ├── text_pipeline.py         # SMS Spam + phishing dataset loader
│   └── graph_pipeline.py        # Elliptic + PaySim graph loaders
├── models/
│   ├── nlp_model.py             # DistilBERT embedder + classifier head
│   ├── nlp_pipeline.py          # NLPPipeline → NLPResult
│   ├── llm_intent.py            # Ollama LLM structured intent extractor
│   ├── gnn_model.py             # GAT model (node + graph level)
│   ├── gnn_pipeline.py          # GNNPipeline → GNNResult
│   ├── fusion_model.py          # Late-fusion MLP
│   ├── fusion_pipeline.py       # FusionPipeline → FusionResult
│   ├── train_nlp.py             # Fine-tune NLP classifier
│   ├── train_gnn.py             # Train GNN on Elliptic dataset
│   └── train_fusion.py          # Train fusion MLP on paired embeddings
├── rag/
│   └── rag_store.py             # ChromaDB vector store + seed patterns
├── explainability/
│   ├── shap_explainer.py        # KernelSHAP on fusion MLP
│   ├── attention_explainer.py   # GAT attention → suspicious neighbors
│   ├── llm_report.py            # LLM-generated compliance report
│   └── explainer.py             # Unified explain() interface
├── api/
│   ├── schemas.py               # Pydantic request/response models
│   └── app.py                   # FastAPI app
└── tests/
    ├── test_nlp.py
    ├── test_gnn.py
    └── test_fusion_api.py
```

---

## Setup

```bash
pip install -r scam_detection/requirements.txt

# Install Ollama and pull Mistral (for LLM intent + reports)
# https://ollama.com
ollama pull mistral
```

---

## Data

### NLP (Text)
Downloaded automatically via HuggingFace `datasets`:
```bash
python -m scam_detection.data.text_pipeline
```

### GNN (Transaction Graph)
Download the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place files in `scam_detection/data/raw/`:
```
elliptic_txs_features.csv
elliptic_txs_edgelist.csv
elliptic_txs_classes.csv
```

---

## Training

Run in order:

```bash
# 1. Fine-tune NLP classifier
python -m scam_detection.models.train_nlp

# 2. Train GNN on Elliptic graph
python -m scam_detection.models.train_gnn

# 3. Train fusion MLP (requires trained NLP + GNN weights)
python -m scam_detection.models.train_fusion
```

---

## Run the API

```bash
uvicorn scam_detection.api.app:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Your SBI account will be blocked. Update KYC now.",
    "nodes": [{"node_id": 0, "features": [49999.0, 2.0, 5.0, 1.0]}],
    "edges": [],
    "target_node": 0,
    "explain": true
  }'
```

### Example Response

```json
{
  "risk_score": 0.94,
  "label": "SCAM",
  "nlp_intent": {
    "intent": "phishing",
    "tactics": ["urgency", "impersonation"],
    "entities": ["SBI", "KYC"],
    "risk_score": 0.91,
    "reason": "Impersonates SBI and creates urgency around account blocking."
  },
  "rag_matches": [
    {"text": "Your KYC is expired...", "label": "kyc_scam", "distance": 0.08}
  ],
  "explanation": {
    "shap_summary": {
      "nlp_total_contribution": 0.62,
      "gnn_total_contribution": 0.38
    },
    "top_neighbors": [],
    "report": "This message impersonates SBI using urgency tactics around KYC expiry..."
  }
}
```

---

## Tests

```bash
pytest scam_detection/tests/ -v
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| DistilBERT over BERT | 40% faster, 97% accuracy — suitable for real-time |
| GAT over GCN | Attention weights are interpretable; highlight suspicious neighbors |
| Late fusion | NLP and GNN independently debuggable; modular training |
| KernelSHAP | Model-agnostic; works on the black-box fusion MLP |
| Ollama (local LLM) | No API cost, no data leaves the machine |
| ChromaDB RAG | Persistent scam pattern memory without retraining |

---

## Datasets

| Module | Dataset | Source |
|---|---|---|
| NLP | SMS Spam Collection | HuggingFace / UCI |
| NLP | Phishing Email Dataset | HuggingFace |
| GNN | Elliptic Bitcoin Dataset | Kaggle |
| GNN | PaySim (alternative) | Kaggle |
