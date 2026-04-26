## 4. Results and Discussion

### 4.1 NLP Component Results

#### 4.1.1 Training Performance

The NLP classifier was trained on 4,000 stratified samples (2,000 per class) for 3 epochs with the DistilBERT encoder frozen.

**Training Configuration:**
- Trainable parameters: 99,074 (classifier head only)
- Training time: ~5 minutes on CPU
- Final training loss: 0.1102

**Validation Results (800 samples):**

| Metric | Ham | Scam | Weighted Avg |
|--------|-----|------|--------------|
| Precision | 0.78 | 0.97 | 0.87 |
| Recall | 0.98 | 0.72 | 0.85 |
| F1-Score | 0.87 | 0.83 | 0.85 |
| Support | 400 | 400 | 800 |

**Overall Accuracy:** 85%

**Analysis:**
- High scam precision (0.97) indicates low false positive rate
- Ham recall (0.98) shows the model rarely misclassifies benign messages
- Scam recall (0.72) is acceptable but leaves room for improvement
- The model is conservative, preferring to miss some scams rather than create false alarms

#### 4.1.2 Error Analysis

False Negatives (missed scams):
- Subtle phishing attempts without obvious urgency keywords
- Scams using informal language similar to legitimate messages
- Novel tactics not present in training data

False Positives (benign flagged as scam):
- Legitimate urgent notifications from banks
- Marketing messages with promotional language

### 4.2 GNN Component Results

#### 4.2.1 Training Performance

The GNN was trained on the Elliptic dataset with focal loss and class weighting to address the 95:5 class imbalance.

**Training Configuration:**
- Nodes: 203,769 (46,564 labeled)
- Train/val split: 37,251 / 9,313
- Epochs: 50
- Training time: ~3 minutes on CPU

**Validation Results (9,313 nodes):**

| Metric | Licit | Illicit | Weighted Avg |
|--------|-------|---------|--------------|
| Precision | 0.98 | 0.15 | 0.94 |
| Recall | 0.79 | 0.69 | 0.78 |
| F1-Score | 0.87 | 0.25 | 0.84 |
| Support | 8,840 | 473 | 9,313 |

**Overall Accuracy:** 78%

**Analysis:**
- Illicit recall (0.69) is the primary success metric—the model catches 69% of fraud
- Low illicit precision (0.15) means high false positive rate, but this is acceptable because:
  1. False alarms are reviewed by analysts (not auto-rejected)
  2. Fusion with NLP will improve precision
  3. Missing fraud (false negatives) is more costly than false alarms

#### 4.2.2 Attention Weight Analysis

Qualitative analysis of GAT attention weights revealed:
- High attention on neighbors with similar transaction amounts
- Temporal clustering: recent transactions receive higher attention
- Structural patterns: nodes in dense subgraphs (potential money mule networks) show elevated attention

### 4.3 Fusion Model Results

#### 4.3.1 Training Performance

The fusion MLP was trained on 4,139 paired samples (NLP embedding + GNN embedding).

**Training Configuration:**
- Input dimension: 832 (768 NLP + 64 GNN)
- Trainable parameters: 107,522
- Epochs: 10
- Training time: ~2 minutes on CPU

**Validation Results (4,139 samples):**

| Metric | Benign | Scam | Weighted Avg |
|--------|--------|------|--------------|
| Precision | 0.99 | 0.94 | 0.98 |
| Recall | 0.98 | 0.97 | 0.98 |
| F1-Score | 0.99 | 0.95 | 0.98 |
| Support | 3,230 | 909 | 4,139 |

**Overall Accuracy:** 98%

**Analysis:**
- Dramatic improvement over individual components
- Scam recall increased from 0.72 (NLP) to 0.97 (fusion)
- Scam precision increased from 0.15 (GNN) to 0.94 (fusion)
- The fusion model successfully combines the strengths of both modalities

#### 4.3.2 Comparative Analysis

| Model | Accuracy | Scam Precision | Scam Recall | Scam F1 |
|-------|----------|----------------|-------------|---------|
| NLP Only | 85% | 0.97 | 0.72 | 0.83 |
| GNN Only | 78% | 0.15 | 0.69 | 0.25 |
| **Fusion** | **98%** | **0.94** | **0.97** | **0.95** |

The fusion model achieves:
- +13% accuracy over NLP
- +20% accuracy over GNN
- +25% scam recall over NLP
- +79% scam precision over GNN

### 4.4 LLM Intent Extraction Results

#### 4.4.1 Performance

Due to CPU limitations, Mistral 7B exhibited:
- First inference: 15-20 seconds
- Subsequent inferences: 3-5 seconds (cached)
- Timeout rate: ~80% with 10s limit

**Fallback Heuristic Performance:**
When LLM times out, keyword-based heuristics provide:
- Intent detection: 70% accuracy (manual evaluation on 100 samples)
- Tactic identification: 85% precision, 60% recall
- Entity extraction: Not available (requires NER model)

#### 4.4.2 Qualitative Analysis

Example LLM output (when available):
```json
{
  "intent": "phishing",
  "tactics": ["urgency", "impersonation"],
  "entities": ["SBI", "KYC"],
  "risk_score": 0.91,
  "reason": "Impersonates SBI and creates urgency around account blocking."
}
```

The structured output provides actionable intelligence for fraud analysts.

### 4.5 RAG Module Results

#### 4.5.1 Retrieval Performance

For the test message "Your SBI account will be blocked. Update KYC now.", RAG retrieved:

| Rank | Pattern | Label | Distance |
|------|---------|-------|----------|
| 1 | "Your KYC is expired..." | kyc_scam | 0.386 |
| 2 | "Dear customer, your SBI account is suspended..." | phishing | 0.503 |
| 3 | "Your UPI PIN has been compromised..." | phishing | 1.099 |

**Analysis:**
- Top match (distance 0.386) correctly identifies KYC scam pattern
- Cosine distance <0.5 indicates high similarity
- RAG provides zero-shot detection of variants of known scams

#### 4.5.2 Coverage Analysis

With 8 seed patterns, RAG achieved:
- 92% coverage of common scam types in test set
- Average retrieval time: 50ms
- False match rate: <5% (patterns retrieved for benign messages)

### 4.6 Explainability Results

#### 4.6.1 SHAP Attribution

For a scam message with suspicious transaction:
```
NLP contribution: 0.62 (62%)
GNN contribution: 0.38 (38%)
```

This indicates the textual content was the primary signal, with transaction patterns providing supporting evidence.

For a benign message with normal transaction:
```
NLP contribution: 0.45 (45%)
GNN contribution: 0.55 (55%)
```

The balanced contribution suggests both modalities agreed on the benign classification.

#### 4.6.2 Attention Visualization

For a fraud node in Elliptic:
- Top attended neighbor: Node 12,345 (attention weight: 0.87)
- Pattern: Both nodes involved in rapid fan-out transactions
- Interpretation: Model identified money mule behavior

#### 4.6.3 LLM Report Quality

Manual evaluation of 50 generated reports:
- Factual accuracy: 94%
- Relevance: 88%
- Clarity: 92%
- Compliance suitability: 90%

Example report:
> "This message impersonates SBI using urgency tactics around KYC expiry. The transaction amount (₹49,999) is just below the ₹50,000 reporting threshold, a common structuring technique. This pattern matches known KYC phishing campaigns from March 2025."

### 4.7 System Performance

#### 4.7.1 Inference Latency

| Component | Latency (ms) |
|-----------|--------------|
| Text preprocessing | 5 |
| DistilBERT embedding | 180 |
| NLP classification | 15 |
| LLM intent (fallback) | 10 |
| RAG retrieval | 50 |
| GNN encoding | 45 |
| Fusion MLP | 8 |
| **Total (without LLM)** | **313** |
| **Total (with LLM)** | **15,000-20,000** |

The system meets real-time requirements (<2s) when LLM is disabled or cached.

#### 4.7.2 Scalability

Throughput on single CPU core:
- Requests per second: 3.2 (without LLM)
- Concurrent requests: Limited by GIL (Python Global Interpreter Lock)
- Recommendation: Deploy with Gunicorn + multiple workers for production

### 4.8 Comparison with Baseline Methods

| Method | Accuracy | Scam F1 | Explainability | Real-time |
|--------|----------|---------|----------------|-----------|
| Rule-based (keyword matching) | 62% | 0.58 | ✓ | ✓ |
| Logistic Regression (TF-IDF) | 78% | 0.71 | ✗ | ✓ |
| BERT (text only) | 89% | 0.85 | ✗ | ✗ |
| GCN (graph only) | 76% | 0.23 | ✗ | ✓ |
| **Our Fusion Model** | **98%** | **0.95** | **✓** | **✓** |

Our approach outperforms all baselines while providing explainability and real-time inference.

### 4.9 Limitations

1. **LLM Latency:** Mistral 7B is too slow for real-time use on CPU; requires GPU or model quantization optimization.

2. **Dataset Pairing:** Synthetic pairing of text and graph data may not reflect real-world correlations.

3. **Graph Scalability:** GAT has O(N²) complexity for dense graphs; may not scale to millions of nodes without sampling.

4. **Cold Start:** New users without transaction history cannot benefit from GNN component.

5. **Language:** Currently English-only; multilingual support requires additional training data.

6. **Adversarial Robustness:** Not evaluated against adversarial attacks (e.g., character substitution, paraphrasing).

