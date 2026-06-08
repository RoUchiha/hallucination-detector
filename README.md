# Hallucination Detector

> A production-grade pipeline for detecting factual hallucinations in LLM-generated text using a dual-scoring approach: Neural Language Inference (NLI) + LLM-as-judge.

---

## What Is This?

Large Language Models (LLMs) like GPT-4 and Claude sometimes generate text that sounds confident and fluent but contains facts that are simply wrong or not supported by any source. These errors are called **hallucinations**.

This pipeline takes two inputs:
1. A **source document** — the ground truth (e.g., a Wikipedia article, a research paper, a knowledge base entry)
2. **LLM-generated text** — a response you want to fact-check against that source

It then:
- Breaks the generated text into individual **atomic claims** (one fact per unit)
- Scores each claim using a **neural NLI model** (Natural Language Inference)
- Escalates uncertain claims to an **LLM judge** for a second opinion
- Outputs a **structured report** with per-claim verdicts and an overall risk score

**Why does this matter?** In production RAG (Retrieval-Augmented Generation) systems, medical AI tools, legal document summarizers, and any system where factual accuracy is critical, hallucination detection is a first-class requirement — not an afterthought.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Hallucination Detection Pipeline               │
│                                                                    │
│   source.txt ──┐                                                   │
│                │                                                   │
│                ▼                                                   │
│   generated.txt ──► [Claim Extractor] ──► List[Claim]             │
│                 (LLM decomposes text       (atomic facts)          │
│                  into atomic claims)                               │
│                              │                                     │
│                              ▼                                     │
│                      [NLI Scorer]                                  │
│                 (DeBERTa cross-encoder)                            │
│                 ENTAILED / NEUTRAL / CONTRADICTED                  │
│                 + confidence score 0.0–1.0                         │
│                              │                                     │
│                    conf < 0.75? ──yes──► [LLM Judge]              │
│                              │           (Claude/GPT second pass)  │
│                              ▼                                     │
│                    [Report Generator]                              │
│                  SUPPORTED / HALLUCINATED / UNCERTAIN              │
│                  Overall hallucination rate % + risk tier          │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **NLI-first, LLM-second** | NLI is fast and cheap (runs locally). LLM calls are reserved for edge cases where NLI confidence is low, keeping API costs minimal. |
| **Chunked source documents** | NLI models have a 512-token limit. Long source docs are split into overlapping chunks; entailment score is the max across all chunks — ensuring no supporting evidence is missed. |
| **Atomic claim decomposition** | Evaluating entire paragraphs against a source is unreliable. Breaking responses into one-fact units gives precise, actionable verdicts. |
| **Pydantic data models** | Every intermediate result is a typed, validated Pydantic model. This makes the pipeline composable, testable, and easy to serialize to JSON. |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Claim Extraction | Claude (Anthropic SDK) |
| NLI Scoring | `cross-encoder/nli-deberta-v3-small` via HuggingFace `transformers` |
| LLM Judge | Claude Haiku (configurable) |
| CLI | `typer` |
| Console output | `rich` |
| Data models | `pydantic` v2 |
| Tests | `pytest` + `pytest-cov` |

---

## Installation

```bash
git clone https://github.com/RoUchiha/hallucination-detector.git
cd hallucination-detector
pip install -e ".[dev]"
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick Start

```bash
# Basic run — NLI only
python -m hallucination_detector \
  --source examples/source.txt \
  --generated examples/generated.txt \
  --output report.json

# With LLM judge for uncertain claims
python -m hallucination_detector \
  --source examples/source.txt \
  --generated examples/generated.txt \
  --output report.json \
  --llm-judge

# Also save an HTML report
python -m hallucination_detector \
  --source examples/source.txt \
  --generated examples/generated.txt \
  --output report.json \
  --html report.html \
  --llm-judge
```

### Example Output

```
Hallucination Detection Report  Risk: HIGH  Rate: 50.0% (3 / 6 claims)

╭───┬────────────────────────────────────────────┬──────────────┬──────┬─────────────┬─────────────╮
│ # │ Claim                                      │ NLI          │ Conf │ LLM         │ Final       │
├───┼────────────────────────────────────────────┼──────────────┼──────┼─────────────┼─────────────┤
│ 1 │ The Eiffel Tower is an iconic iron struct… │ ENTAILED     │ 0.91 │ —           │ SUPPORTED   │
│ 2 │ It stands approximately 330 metres tall.   │ ENTAILED     │ 0.88 │ —           │ SUPPORTED   │
│ 3 │ It was built in 1885.                      │ CONTRADICTED │ 0.94 │ unsupported │ HALLUCINATED│
│ 4 │ The tower was made of steel, not iron.     │ CONTRADICTED │ 0.89 │ unsupported │ HALLUCINATED│
╰───┴────────────────────────────────────────────┴──────────────┴──────┴─────────────┴─────────────╯
```

---

## Running Tests

```bash
pytest --cov=src/hallucination_detector --cov-report=term-missing
```

All LLM and NLI calls are mocked in tests — no API key or GPU required for CI.

---

## Report Schema

```json
{
  "source_doc_hash": "a3f8c2b1...",
  "generated_text_hash": "d9e1f4a2...",
  "total_claims": 6,
  "hallucination_rate": 0.5,
  "risk_tier": "HIGH",
  "generated_at": "2025-01-01T12:00:00Z",
  "verdicts": [
    {
      "claim": { "id": 1, "text": "...", "span_start": 0, "span_end": 45 },
      "nli_label": "CONTRADICTED",
      "nli_confidence": 0.94,
      "llm_verdict": "unsupported",
      "llm_reasoning": "The source states 1889, not 1885.",
      "final_verdict": "HALLUCINATED"
    }
  ]
}
```

---

## Project Structure

```
hallucination-detector/
├── src/hallucination_detector/
│   ├── cli.py           # typer CLI entrypoint
│   ├── extractor.py     # LLM-based claim decomposition
│   ├── nli_scorer.py    # DeBERTa NLI with chunking
│   ├── llm_judge.py     # LLM escalation pass
│   ├── reporter.py      # rich console + JSON + HTML output
│   └── models.py        # Pydantic data models
├── tests/               # pytest suite with mocked LLM/NLI
├── examples/            # sample source + generated text
└── pyproject.toml
```

---

## Extending This

- **Swap the NLI model**: Change `--nli-model` to any cross-encoder on HuggingFace
- **Swap the LLM judge**: Replace the Anthropic client in `llm_judge.py` with OpenAI or any other provider
- **Add a FastAPI wrapper**: Mount the pipeline behind `POST /detect` for production use
- **Add FAISS retrieval**: Before NLI scoring, retrieve the top-k most semantically similar source chunks using a vector index (see `stretch/faiss_retrieval.py` for a starter)

---

## Ethics Note

This tool is designed for **defensive use**: catching factual errors before they reach users. It does not generate harmful content. Hallucination detection is a critical safety layer for any production LLM deployment.
