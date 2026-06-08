"""NLI-based hallucination scoring using DeBERTa cross-encoder."""

from __future__ import annotations

from typing import List, Tuple

from transformers import pipeline

from .models import Claim, ClaimVerdict

# Label mapping from the cross-encoder model's output
_LABEL_MAP = {
    "entailment": "ENTAILED",
    "neutral": "NEUTRAL",
    "contradiction": "CONTRADICTED",
}

# Threshold below which we escalate to LLM judge
LLM_ESCALATION_THRESHOLD = 0.75


class NLIScorer:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: int = -1):
        # device=-1 → CPU; set to 0 for CUDA
        self.pipe = pipeline("zero-shot-classification", model=model_name, device=device)
        self._chunk_size = 450  # tokens (conservative, model max is 512)

    def _chunk_text(self, text: str, words_per_chunk: int = 300) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i : i + words_per_chunk])
            for i in range(0, len(words), words_per_chunk)
        ]

    def _score_claim_against_chunk(self, claim_text: str, chunk: str) -> Tuple[str, float]:
        result = self.pipe(
            chunk,
            candidate_labels=[claim_text],
            hypothesis_template="{}",
            multi_label=False,
        )
        # zero-shot-classification wraps NLI internally; use raw NLI pipeline instead
        return result["labels"][0], result["scores"][0]

    def score_claim(self, claim: Claim, source_doc: str) -> Tuple[str, float]:
        """Return (nli_label, confidence) — takes max entailment across chunks."""
        chunks = self._chunk_text(source_doc)
        best_label = "NEUTRAL"
        best_conf = 0.0

        for chunk in chunks:
            # Use raw NLI: premise=chunk, hypothesis=claim
            result = self.pipe(
                sequences=chunk,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template=f"Based on the text: {claim.text}",
                multi_label=False,
            )
            label_raw = result["labels"][0].lower()
            conf = result["scores"][0]

            # Prioritise entailment; also track highest-confidence contradiction
            if label_raw == "entailment" and conf > best_conf:
                best_label = "ENTAILED"
                best_conf = conf
            elif label_raw == "contradiction" and best_label != "ENTAILED" and conf > best_conf:
                best_label = "CONTRADICTED"
                best_conf = conf
            elif label_raw == "neutral" and best_label == "NEUTRAL" and conf > best_conf:
                best_conf = conf

        return best_label, best_conf

    def score_all(self, claims: List[Claim], source_doc: str) -> List[ClaimVerdict]:
        verdicts = []
        for claim in claims:
            label, conf = self.score_claim(claim, source_doc)
            # Determine provisional final verdict; LLM judge may override low-confidence
            if label == "ENTAILED" and conf >= LLM_ESCALATION_THRESHOLD:
                final = "SUPPORTED"
            elif label == "CONTRADICTED":
                final = "HALLUCINATED"
            else:
                final = "UNCERTAIN"

            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    nli_label=label,
                    nli_confidence=conf,
                    final_verdict=final,
                )
            )
        return verdicts
