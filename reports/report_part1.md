# Early Detection of Financial Scams and Fraudulent Messages Using Context-Aware NLP and Graph-Based Learning

**Authors:**  
Malhar Shinde  25070149014
Jay Chauhan    25070149006

**Affiliation:**  
Symbiosis Institute of Technology, Pune Campus  
Symbiosis International (Deemed University)  
Pune, India

---

## Abstract

Financial fraud and scam messages have become increasingly sophisticated, causing significant economic losses globally. Traditional rule-based detection systems struggle to adapt to evolving fraud patterns and lack the ability to analyze both textual content and transactional relationships simultaneously. This project presents a novel multi-modal framework that integrates Natural Language Processing (NLP) and Graph Neural Networks (GNN) for real-time detection of financial scams. The system employs DistilBERT for semantic text analysis, Graph Attention Networks (GAT) for transaction pattern recognition, and a Large Language Model (LLM) for intent extraction. A Retrieval-Augmented Generation (RAG) component provides pattern memory by matching incoming messages against known scam databases. The fusion model combines embeddings from both modalities to achieve 98% accuracy on paired text-transaction data. Individual components achieved 85% accuracy (NLP) and 69% recall (GNN) on their respective tasks. The system provides explainability through SHAP analysis and attention weight visualization, making it suitable for regulatory compliance. Deployed as a FastAPI service with a Streamlit interface, the framework demonstrates practical applicability for real-time fraud detection in banking and payment systems.

**Keywords:** Financial Fraud Detection, Natural Language Processing, Graph Neural Networks, Multi-Modal Learning, Explainable AI, Retrieval-Augmented Generation

---

## 1. Introduction

### 1.1 Background

The proliferation of digital payment systems and online banking has revolutionized financial transactions, but it has also created new opportunities for fraudulent activities. Financial scams manifest in multiple forms: phishing emails impersonating banks, SMS messages creating urgency around KYC updates, lottery scams, and sophisticated social engineering attacks. According to the Reserve Bank of India, digital payment fraud cases increased by 300% between 2020 and 2023, with losses exceeding ₹1,200 crores annually.

Traditional fraud detection systems rely on rule-based approaches or simple machine learning classifiers that analyze either textual content or transaction patterns in isolation. However, modern scams exploit both channels simultaneously—a fraudulent message may accompany a suspicious transaction pattern, and neither signal alone provides sufficient evidence for detection.

### 1.2 Motivation

The limitations of existing approaches motivated this research:

1. **Unimodal Analysis:** Most systems analyze text or transactions separately, missing correlations between message content and behavioral patterns.

2. **Static Pattern Matching:** Rule-based systems cannot adapt to novel scam tactics without manual updates.

3. **Lack of Explainability:** Black-box models provide predictions without justification, making them unsuitable for regulatory compliance and fraud analyst workflows.

4. **Cold Start Problem:** New scam patterns require extensive labeled data before detection becomes effective.

5. **Real-Time Requirements:** Banking systems need sub-second response times, which many deep learning approaches cannot satisfy.

### 1.3 Problem Statement

Given a financial message (SMS, email, or payment note) and associated transaction metadata, the system must:

1. Classify the message as SCAM or SAFE with high precision and recall
2. Identify manipulation tactics and extracted entities
3. Detect anomalous transaction patterns in the user's behavioral graph
4. Provide human-interpretable explanations for predictions
5. Operate in real-time (<2 seconds per prediction)
6. Adapt to new scam patterns without full model retraining

### 1.4 Objectives

The primary objectives of this project are:

1. **Multi-Modal Integration:** Develop a framework that jointly analyzes textual and graph-structured data for fraud detection.

2. **High Performance:** Achieve >90% accuracy while maintaining >85% recall on scam detection.

3. **Explainability:** Provide feature attribution, attention visualization, and natural language explanations for each prediction.

4. **Adaptability:** Implement RAG-based pattern memory that can incorporate new scam examples without retraining.

5. **Production Readiness:** Deploy as a scalable API with monitoring, logging, and compliance features.

### 1.5 Contributions

This work makes the following contributions:

1. **Novel Architecture:** First framework to combine DistilBERT, GAT, and LLM-based intent extraction in a unified pipeline for financial fraud detection.

2. **Late Fusion Strategy:** Demonstrates that training NLP and GNN components independently before fusion improves modularity and debuggability.

3. **RAG Integration:** Shows that retrieval-augmented generation can provide zero-shot adaptation to new scam patterns.

4. **Explainability Framework:** Combines SHAP, attention weights, and LLM-generated reports for multi-level interpretability.

5. **Open-Source Implementation:** Provides a complete, reproducible codebase with training scripts, API, and UI.

### 1.6 Report Organization

The remainder of this report is organized as follows: Section 2 reviews related work in fraud detection, NLP, and GNN applications. Section 3 describes the methodology, including dataset preparation, model architectures, and training procedures. Section 4 presents experimental results and comparative analysis. Section 5 discusses findings, limitations, and implications. Section 6 concludes with future research directions.

