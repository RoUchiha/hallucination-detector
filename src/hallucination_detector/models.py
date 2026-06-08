"""Pydantic data models for the hallucination detection pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
import hashlib

from pydantic import BaseModel, field_validator


class Claim(BaseModel):
    id: int
    text: str
    span_start: int
    span_end: int


class ClaimVerdict(BaseModel):
    claim: Claim
    nli_label: Literal["ENTAILED", "NEUTRAL", "CONTRADICTED"]
    nli_confidence: float
    llm_verdict: Optional[Literal["supported", "unsupported", "unknown"]] = None
    llm_reasoning: Optional[str] = None
    final_verdict: Literal["SUPPORTED", "HALLUCINATED", "UNCERTAIN"]

    @field_validator("nli_confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("nli_confidence must be between 0 and 1")
        return v


class HallucinationReport(BaseModel):
    source_doc_hash: str
    generated_text_hash: str
    total_claims: int
    hallucination_rate: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    verdicts: List[ClaimVerdict]
    generated_at: datetime

    @classmethod
    def compute_hash(cls, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
