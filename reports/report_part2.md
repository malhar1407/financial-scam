## 2. Literature Review

### 2.1 Traditional Fraud Detection Methods

Early fraud detection systems relied on rule-based approaches and statistical anomaly detection. Bolton and Hand (2002) surveyed statistical methods including logistic regression, decision trees, and neural networks for credit card fraud detection. While effective for known patterns, these methods struggled with novel attack vectors and required extensive feature engineering.

Phua et al. (2010) proposed a comprehensive framework for fraud detection in telecommunications, demonstrating that ensemble methods combining multiple weak classifiers could improve detection rates. However, their approach required domain-specific feature extraction and could not generalize across fraud types.

### 2.2 Natural Language Processing for Fraud Detection

The application of NLP to fraud detection gained prominence with the rise of phishing and social engineering attacks. Fette et al. (2007) developed PILFER, a machine learning-based phishing email classifier using lexical and structural features. Their system achieved 96% accuracy but relied on hand-crafted features that required regular updates.

Devlin et al. (2019) introduced BERT (Bidirectional Encoder Representations from Transformers), revolutionizing NLP through pre-trained contextual embeddings. Subsequent work by Sanh et al. (2019) on DistilBERT demonstrated that knowledge distillation could reduce model size by 40% while retaining 97% of BERT's performance, making it suitable for production deployments.

Recent studies have applied transformer models to fraud detection. Hiransha et al. (2021) used BERT for SMS spam classification, achieving 94% accuracy on the UCI SMS Spam dataset. However, their approach did not incorporate transaction context or provide explainability.

### 2.3 Graph Neural Networks for Financial Fraud

Graph-based approaches model financial systems as networks where nodes represent accounts and edges represent transactions. Kipf and Welling (2017) introduced Graph Convolutional Networks (GCN), enabling deep learning on graph-structured data. Their work inspired numerous applications in fraud detection.

Weber et al. (2019) released the Elliptic Bitcoin dataset, containing 203,769 Bitcoin transactions with fraud labels. They demonstrated that GCNs could detect illicit transactions with 94% accuracy by leveraging network structure. However, their model suffered from class imbalance (only 2% fraud) and did not incorporate textual information.

Veličković et al. (2018) proposed Graph Attention Networks (GAT), which learn importance weights for neighboring nodes. Liu et al. (2021) applied GAT to credit card fraud detection, showing that attention mechanisms improve interpretability by highlighting suspicious transaction patterns.

### 2.4 Multi-Modal Learning

Multi-modal learning combines information from heterogeneous sources. Baltrusaitis et al. (2019) surveyed fusion strategies, categorizing them as early fusion (feature-level), late fusion (decision-level), and hybrid approaches. Late fusion has proven effective when modalities have different optimal architectures.

In fraud detection, Pourhabibi et al. (2020) combined transaction features with user behavior logs using a two-stream neural network. Their late fusion approach achieved 91% F1-score on credit card fraud, demonstrating the value of multi-modal analysis. However, they did not incorporate textual data or provide explainability.

### 2.5 Explainable AI in Fraud Detection

Regulatory requirements (e.g., GDPR, RBI guidelines) mandate explainability in automated decision systems. Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), a unified framework for interpreting model predictions based on game theory. SHAP has been successfully applied to fraud detection by Bhattacharyya et al. (2011), who demonstrated that feature attribution improves fraud analyst productivity.

Attention mechanisms in neural networks provide inherent interpretability. Bahdanau et al. (2015) showed that attention weights reveal which input elements influenced predictions. In graph networks, attention weights indicate which neighbors contributed to node classifications, enabling fraud investigators to trace suspicious connections.

### 2.6 Large Language Models and RAG

Recent advances in Large Language Models (LLMs) have enabled structured information extraction from unstructured text. Brown et al. (2020) demonstrated that GPT-3 could perform few-shot learning on diverse NLP tasks. Touvron et al. (2023) released Llama 2, an open-source LLM suitable for local deployment.

Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), combines neural retrieval with generation to incorporate external knowledge without retraining. Gao et al. (2023) applied RAG to cybersecurity threat detection, showing that vector databases of known attack patterns improved zero-shot detection of novel threats.

### 2.7 Research Gaps

Despite significant progress, existing work has limitations:

1. **Isolated Modalities:** Most systems analyze text or graphs separately, missing cross-modal correlations.

2. **Limited Explainability:** Few systems provide multi-level explanations suitable for regulatory compliance.

3. **Static Models:** Retraining is required to adapt to new fraud patterns, causing detection delays.

4. **Benchmark Limitations:** No standard dataset combines textual messages with transaction graphs.

5. **Production Gaps:** Academic systems often lack real-time inference capabilities and deployment considerations.

This project addresses these gaps by proposing a unified framework that integrates NLP, GNN, LLM-based intent extraction, and RAG-powered adaptability, with comprehensive explainability and production-ready deployment.

