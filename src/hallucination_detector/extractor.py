"""Claim extraction from generated text using an LLM."""

from __future__ import annotations

import json
import os
from typing import List

import anthropic

from .models import Claim

SYSTEM_PROMPT = """You are a precise claim extractor. Given a piece of generated text, decompose it into
atomic, self-contained factual claims — one per sentence. Each claim must be independently verifiable.

Respond ONLY with a JSON array:
[
  {"id": 1, "text": "<claim>", "span_start": <int>, "span_end": <int>},
  ...
]

Rules:
- Each claim is a single declarative sentence.
- span_start / span_end are character offsets into the original generated_text.
- Do not invent claims not present in the text.
- Do not include opinions, instructions, or meta-commentary as claims."""


class ClaimExtractor:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def extract(self, generated_text: str) -> List[Claim]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract all atomic factual claims from this text:\n\n{generated_text}",
                }
            ],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return [Claim(**item) for item in data]
