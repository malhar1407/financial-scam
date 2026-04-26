### 3.6 Graph Neural Network Pipeline

#### 3.6.1 Graph Construction

From transaction data, we construct a directed graph G = (V, E) where:
- V: Set of accounts (nodes)
- E: Set of transactions (edges)
- Node features: [amount, hour, day_of_week, is_new_account] (4 dimensions for API, 166 for Elliptic)
- Edge features: Not used (future work)

For Elliptic dataset:
- Nodes represent Bitcoin transactions
- Edges represent flow of bitcoins between transactions
- Features include transaction statistics and aggregated neighbor features

#### 3.6.2 Graph Attention Network (GAT)

Architecture:
```
Input: Node features X ∈ R^(N×F), Edge index E
    ↓
GATConv(F → 64, heads=4, concat=False)
    ↓
ELU + Dropout(0.3)
    ↓
GATConv(64 → 64, heads=4, concat=False)
    ↓
Output: Node embeddings H ∈ R^(N×64)
```

Attention mechanism:
For each node i and neighbor j:
```
α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
h'_i = σ(Σ_j α_ij W h_j)
```

Where:
- α_ij: Attention coefficient (importance of neighbor j to node i)
- W: Learnable weight matrix
- a: Attention vector
- ||: Concatenation
- σ: Activation function (ELU)

Rationale for GAT over GCN:
1. Attention weights provide interpretability
2. Adaptive neighbor weighting improves performance on imbalanced graphs
3. Multi-head attention captures diverse relationship patterns

#### 3.6.3 GNN Training

Challenges:
- Class imbalance: 95% licit, 5% illicit in Elliptic
- Solution: Focal loss with class weights

Focal loss:
```
FL(p_t) = -(1 - p_t)^γ log(p_t)
```
Where γ=2 down-weights easy examples, focusing on hard minority class.

Class weights:
```
w_illicit = n_licit / n_illicit ≈ 18.7
w_licit = 1.0
```

Training configuration:
- Optimizer: Adam (lr=5e-3, weight_decay=1e-4)
- Scheduler: StepLR (step_size=20, gamma=0.5)
- Epochs: 50
- Loss: Focal loss with class weights
- Best checkpoint: Highest illicit F1 on validation set

Results:
- Illicit recall: 69%
- Illicit precision: 15%
- Overall accuracy: 78%

The low precision is acceptable because:
1. Recall is prioritized in fraud detection (missing fraud is worse than false alarms)
2. Fusion with NLP improves precision
3. Human analysts review flagged cases

### 3.7 Fusion Model

#### 3.7.1 Late Fusion Strategy

We employ late fusion rather than early fusion because:
1. NLP and GNN have different optimal architectures
2. Modular training enables independent debugging
3. Components can be updated separately
4. Allows fallback to single modality when one is unavailable

#### 3.7.2 Fusion MLP Architecture

```
Input: Concatenated embeddings [NLP(768) || GNN(64)] = 832 dimensions
    ↓
Linear(832 → 128) + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + ReLU
    ↓
Linear(64 → 2)
    ↓
Output: [2] logits → Softmax → Risk score
```

#### 3.7.3 Fusion Training

Data preparation:
1. Freeze pre-trained NLP and GNN encoders
2. Extract embeddings for all samples
3. Pair embeddings by label (scam text ↔ fraud node, ham text ↔ licit node)
4. Train only the fusion MLP

Configuration:
- Optimizer: Adam (lr=1e-3)
- Loss: Cross-entropy
- Epochs: 10
- Batch size: 32
- Train/val split: 80/20

Paired dataset statistics:
- Total: 4,139 samples
- Benign: 3,230 (78%)
- Scam: 909 (22%)

### 3.8 Explainability Framework

#### 3.8.1 SHAP Analysis

We use KernelSHAP (Lundberg & Lee, 2017) to attribute predictions to input features:

```
φ_i = Σ_{S⊆F\{i}} (|S|!(|F|-|S|-1)!)/(|F|!) [f(S∪{i}) - f(S)]
```

Where:
- φ_i: SHAP value for feature i
- F: Set of all features
- S: Subset of features
- f: Model prediction function

Implementation:
- Background samples: 10 zero vectors (neutral baseline)
- Samples per explanation: 100
- Output: NLP contribution vs GNN contribution

#### 3.8.2 Attention Weight Visualization

GAT attention weights α_ij indicate which neighbors influenced the prediction. For explainability:

1. Extract attention weights from all GAT layers
2. Compute mean attention per neighbor across layers
3. Rank neighbors by attention score
4. Return top-5 most attended neighbors

This reveals suspicious transaction patterns (e.g., money mule networks, circular flows).

#### 3.8.3 LLM-Generated Reports

For regulatory compliance, we generate natural language explanations:

Input to LLM:
```json
{
  "message": "...",
  "risk_score": 0.94,
  "detected_intent": "phishing",
  "tactics": ["urgency", "impersonation"],
  "similar_known_scam": "kyc_scam",
  "nlp_contribution": 0.62,
  "gnn_contribution": 0.38,
  "suspicious_neighbors": [123, 456]
}
```

Output: 3-sentence compliance report explaining the decision.

### 3.9 Deployment Architecture

#### 3.9.1 FastAPI Backend

Endpoints:
- `POST /predict`: Main inference endpoint
- `GET /health`: Liveness check

Features:
- Async request handling
- Automatic model loading at startup
- Graceful error handling
- Request/response validation via Pydantic

#### 3.9.2 Streamlit Frontend

User interface components:
- Message input text area
- Transaction feature sliders
- Real-time prediction display
- Risk score visualization
- Intent analysis panel
- RAG match display
- Example message buttons

#### 3.9.3 System Requirements

Hardware:
- CPU: 4+ cores recommended
- RAM: 8GB minimum (16GB for LLM)
- Storage: 10GB (models + data)

Software:
- Python 3.12
- PyTorch 2.2.2
- PyTorch Geometric 2.7.0
- Transformers 4.39.3
- Ollama (for LLM)

Inference latency:
- NLP: ~200ms
- GNN: ~50ms (single node)
- Fusion: ~10ms
- LLM (optional): ~15s (first call), ~3s (cached)
- Total: <2s without LLM, <20s with LLM

### 3.10 Evaluation Metrics

#### 3.10.1 Classification Metrics

For binary classification (SCAM vs SAFE):

**Precision:**
```
P = TP / (TP + FP)
```

**Recall:**
```
R = TP / (TP + FN)
```

**F1-Score:**
```
F1 = 2 × (P × R) / (P + R)
```

**Accuracy:**
```
Acc = (TP + TN) / (TP + TN + FP + FN)
```

Where:
- TP: True Positives (correctly identified scams)
- TN: True Negatives (correctly identified safe messages)
- FP: False Positives (safe messages flagged as scams)
- FN: False Negatives (missed scams)

#### 3.10.2 Evaluation Strategy

We evaluate at three levels:

1. **Component-level:**
   - NLP classifier on text dataset
   - GNN classifier on Elliptic graph
   - Metrics: Precision, Recall, F1, Accuracy

2. **Fusion-level:**
   - Combined model on paired dataset
   - Metrics: Same as above
   - Comparison with individual components

3. **End-to-end:**
   - API response time
   - Explainability quality (qualitative)
   - User interface usability

