"""Unified NLP pipeline: embedding + LLM intent + RAG retrieval."""
import torch
from dataclasses import dataclass
from scam_detection.models.nlp_model import NLPClassifier
from scam_detection.models.llm_intent import extract_intent
from scam_detection.rag.rag_store import retrieve_similar


@dataclass
class NLPResult:
    embedding: torch.Tensor       # [768] — fed into fusion model
    llm_intent: dict               # structured output from LLM
    rag_matches: list[dict]        # similar known scam patterns
    nlp_logits: torch.Tensor       # [2] — standalone NLP classifier output


class NLPPipeline:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.classifier = NLPClassifier().to(device)
        self.classifier.eval()

    def run(self, text: str) -> NLPResult:
        # 1. DistilBERT embedding + classifier logits
        with torch.no_grad():
            logits = self.classifier([text])          # [1, 2]
            emb = self.classifier.embed([text])[0]    # [768]

        # 2. LLM structured intent extraction
        intent = extract_intent(text)

        # 3. RAG: retrieve similar known scam patterns
        matches = retrieve_similar(text)

        return NLPResult(
            embedding=emb,
            llm_intent=intent,
            rag_matches=matches,
            nlp_logits=logits[0],
        )

    def load_weights(self, path: str):
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
