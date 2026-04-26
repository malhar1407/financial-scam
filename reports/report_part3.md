## 3. Methodology

### 3.1 System Architecture

The proposed framework consists of five major components operating in a pipeline:

1. **NLP Pipeline:** Text preprocessing, DistilBERT embedding, and classification
2. **LLM Intent Extractor:** Structured analysis of manipulation tactics using Mistral 7B
3. **RAG Module:** Vector similarity search against known scam patterns using ChromaDB
4. **GNN Pipeline:** Graph construction, GAT encoding, and node classification
5. **Fusion Layer:** Late fusion MLP combining NLP and GNN embeddings

Figure 1 illustrates the complete architecture:

```
Input: Message + Transaction Graph
        │
        ├──► Text Preprocessor ──► DistilBERT ──► [768-dim embedding]
        │                                └──► NLP Classifier [2 logits]
        │
        ├──► LLM (Mistral 7B) ──► {intent, tactics, entities, risk_score}
        │
        ├──► RAG (ChromaDB) ──► top-k similar scam patterns
        │
        ├──► Graph Builder ──► GAT (4-head attention) ──► [64-dim node embedding]
        │                                └──► GNN Classifier [2 logits]
        │
        └──► Fusion MLP (832 → 128 → 64 → 2)
                    │
                    └──► Risk Score (0–1) + Label (SCAM/SAFE)
                    │
                    └──► Explainability Module
                            ├── SHAP: Feature attribution
                            ├── Attention: Suspicious neighbors
                            └── LLM: Natural language report
```

### 3.2 Dataset Description

Due to the absence of a unified dataset containing both textual messages and transaction graphs, we employed a multi-source strategy:

#### 3.2.1 Text Data

**SMS Spam Collection (UCI Repository)**
- Source: Almeida et al. (2011)
- Size: 5,574 messages
- Labels: ham (4,827), spam (747)
- Language: English
- Characteristics: Short messages (10-160 characters), informal language

**Phishing Email Dataset**
- Source: Liu et al. (2022) via Hugging Face
- Size: 18,650 emails
- Labels: Safe Email (9,325), Phishing Email (9,325)
- Characteristics: Longer text, formal impersonation attempts

**Combined Dataset:**
- Total: 24,224 samples after deduplication
- Train/Val split: 80/20 stratified
- Preprocessing: lowercasing, URL/phone masking, punctuation normalization

#### 3.2.2 Graph Data

**Elliptic Bitcoin Dataset**
- Source: Weber et al. (2019)
- Nodes: 203,769 Bitcoin transactions
- Edges: 234,355 transaction flows
- Features: 166 per node (93 local, 72 aggregated, 1 time step)
- Labels: licit (42,019), illicit (4,545), unknown (157,205)
- Characteristics: Highly imbalanced (2% fraud), temporal evolution

**PaySim (Alternative)**
- Source: Lopez-Rojas et al. (2016)
- Synthetic mobile money transactions
- Used for validation experiments

#### 3.2.3 Synthetic Pairing

Since no dataset contains both messages and graphs, we created synthetic pairs for fusion training:
- Scam messages paired with illicit transaction nodes
- Benign messages paired with licit transaction nodes
- Stratified sampling to maintain class balance
- Final paired dataset: 4,139 samples (3,230 benign, 909 scam)

### 3.3 Text Processing Pipeline

#### 3.3.1 Preprocessing

Text cleaning steps:
1. Convert to lowercase
2. Replace URLs with `<URL>` token
3. Replace phone numbers (10+ digits) with `<PHONE>` token
4. Remove special characters except angle brackets
5. Normalize whitespace

```python
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\b\d{10,}\b", "<PHONE>", text)
    text = re.sub(r"[^\w\s<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
```

#### 3.3.2 DistilBERT Embedder

Architecture:
- Base model: `distilbert-base-uncased` (66M parameters)
- Input: Tokenized text (max 128 tokens)
- Output: [CLS] token embedding (768 dimensions)
- Frozen encoder during training (only classifier head trained)

Rationale for DistilBERT:
- 40% smaller than BERT-base
- 60% faster inference
- 97% of BERT's performance retained
- Suitable for CPU deployment

#### 3.3.3 NLP Classifier Head

Architecture:
```
Input: [768] embedding
    ↓
Linear(768 → 128) + ReLU + Dropout(0.2)
    ↓
Linear(128 → 2)
    ↓
Output: [2] logits (ham, scam)
```

Training:
- Optimizer: AdamW (lr=1e-3)
- Loss: Cross-entropy
- Epochs: 3
- Batch size: 32
- Trainable parameters: 99,074 (encoder frozen)

### 3.4 LLM Intent Extraction

#### 3.4.1 Model Selection

We use Mistral 7B (Jiang et al., 2023) via Ollama for local deployment:
- Parameters: 7 billion
- Context window: 8,192 tokens
- Quantization: 4-bit for CPU inference
- Deployment: Local (no data leaves the system)

#### 3.4.2 Structured Prompting

System prompt:
```
You are a financial scam detection expert.
Analyze the message and return ONLY valid JSON with these fields:
- intent: one of [phishing, impersonation, urgency_scam, lottery_scam, benign]
- tactics: list of manipulation tactics (e.g., urgency, authority, fear)
- entities: extracted entities (bank names, amounts, URLs, phone numbers)
- risk_score: float 0.0-1.0 (confidence this is a scam)
- reason: one sentence explanation
```

Fallback mechanism:
- Timeout: 10 seconds
- If LLM unavailable: heuristic-based intent detection using keyword matching
- Ensures system remains operational even without LLM

### 3.5 Retrieval-Augmented Generation (RAG)

#### 3.5.1 Vector Store

Technology: ChromaDB with persistent storage
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Distance metric: Cosine similarity
- Top-k retrieval: 3 most similar patterns

#### 3.5.2 Seed Patterns

Initial knowledge base (8 patterns):
1. KYC expiry scam
2. Lottery/prize scam
3. Bank account suspension phishing
4. Unauthorized login alert
5. Cashback activation urgency
6. UPI PIN compromise
7. Income tax refund phishing
8. Work-from-home job scam

Extensibility: New patterns can be added without retraining via `rag_store.add()`.

