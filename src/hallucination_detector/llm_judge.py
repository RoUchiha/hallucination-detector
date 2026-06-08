"""LLM-as-judge escalation for low-confidence NLI claims."""

from __future__ import annotations

import json
import os
from typing import List

import anthropic

from .models import ClaimVerdict
from .nli_scorer import LLM_ESCALATION_THRESHOLD

JUDGE_SYSTEM = """You are a rigorous fact-checker. Given a source document and a claim extracted from an
LLM-generated response, determine whether the claim is supported by the source document.

Respond ONLY with valid JSON in this exact format:
{"verdict": "supported" | "unsupported" | "unknown", "reasoning": "<one sentence explanation>"}

Rules:
- "supported": the source document explicitly or strongly implies the claim.
- "unsupported": the source document contradicts or does not mention the claim.
- "unknown": the source document does not contain enough information to judge."""


class LLMJudge:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def judge_claim(self, claim_text: str, source_doc: str) -> tuple[str, str]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"SOURCE DOCUMENT:\n{source_doc[:3000]}\n\n"
                        f"CLAIM TO VERIFY:\n{claim_text}"
                    ),
                }
            ],
        )
        raw = message.content[0].text.strip()
        data = json.loads(raw)
        return data["verdict"], data["reasoning"]

    def escalate(self, verdicts: List[ClaimVerdict], source_doc: str) -> List[ClaimVerdict]:
        """Run LLM judge on claims where NLI confidence is below threshold."""
        updated = []
        for v in verdicts:
            if v.nli_confidence < LLM_ESCALATION_THRESHOLD or v.final_verdict == "UNCERTAIN":
                llm_verdict, reasoning = self.judge_claim(v.claim.text, source_doc)
                final = (
                    "SUPPORTED" if llm_verdict == "supported"
                    else "HALLUCINATED" if llm_verdict == "unsupported"
                    else "UNCERTAIN"
                )
                updated.append(
                    v.model_copy(
                        update={
                            "llm_verdict": llm_verdict,
                            "llm_reasoning": reasoning,
                            "final_verdict": final,
                        }
                    )
                )
            else:
                updated.append(v)
        return updated
