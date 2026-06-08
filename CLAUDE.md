# CLAUDE.md — Hallucination Detector

This file gives Claude Code full context for this project. Read it before making any changes.

---

## What This Project Does

A two-stage hallucination detection pipeline that takes a **source document** (ground truth) and **LLM-generated text**, decomposes the generated text into atomic claims, scores each claim for factual support, and outputs a structured risk report.

**Stage 1 — NLI Scorer**: Runs each claim against the source document using the `cross-encoder/nli-deberta-v3-small` transformer. Labels each claim ENTAILED / NEUTRAL / CONTRADICTED with a float confidence. Source docs longer than ~300 words are chunked; max entailment score across chunks is used.

**Stage 2 — LLM Judge (optional)**: Claims where NLI confidence < 0.75 are escalated to Claude for a second-opinion verdict (`supported` / `unsupported` / `unknown`). The `--llm-judge` CLI flag enables this.

Final verdicts: `SUPPORTED` / `HALLUCINATED` / `UNCERTAIN`. Report includes overall hallucination rate % and risk tier `LOW / MEDIUM / HIGH`.

---

## Repository Layout

```
hallucination-detector/
├── src/hallucination_detector/
│   ├── models.py        # Pydantic: Claim, ClaimVerdict, HallucinationReport
│   ├── extractor.py     # LLM claim decomposition (Claude API)
│   ├── nli_scorer.py    # DeBERTa NLI with chunking logic
│   ├── llm_judge.py     # LLM escalation for low-confidence claims
│   ├── reporter.py      # rich console + JSON + HTML output
│   ├── cli.py           # typer CLI (--source, --generated, --output, --llm-judge)
│   └── __main__.py      # python -m hallucination_detector entrypoint
├── tests/
│   ├── test_extractor.py    # mocked Anthropic client
│   ├── test_nli_scorer.py   # mocked transformers pipeline
│   └── test_reporter.py     # report math + JSON roundtrip
├── examples/
│   ├── source.txt       # Eiffel Tower facts (ground truth)
│   └── generated.txt    # contains injected hallucinations
└── pyproject.toml
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM calls | `anthropic` SDK |
| NLI model | `transformers` (`cross-encoder/nli-deberta-v3-small`) |
| Data models | `pydantic` v2 |
| CLI | `typer` |
| Console output | `rich` |
| Tests | `pytest`, `pytest-cov` |

---

## Environment

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

```bash
# Run end-to-end
python -m hallucination_detector --source examples/source.txt --generated examples/generated.txt --output report.json --llm-judge

# Run tests (no API key needed — all LLM/NLI calls mocked)
pytest

# Type check
python -c "from hallucination_detector import *"
```

---

## Key Design Decisions

- **NLI-first**: local model, no API cost, handles clear-cut cases (high confidence entailment/contradiction)
- **LLM-second**: only called when NLI confidence < 0.75 — minimizes API spend
- **Chunking in NLIScorer**: `_chunk_text()` splits source into ~300-word windows, takes max entailment score
- **Pydantic everywhere**: `Claim → ClaimVerdict → HallucinationReport` — typed, serializable, testable
- **Exit codes**: CLI exits 0 on success regardless of hallucination rate (rate is in report, not exit code)

---

## Course Context

Built as part of the **UT Austin AI & Machine Learning** program (McCombs, 23-week executive program).
- **Course 03** — Generative AI for NLP: LLM APIs, responsible AI, RAG chunking pattern
- **HuggingFace** tooling from the curriculum's 30+ tool coverage

---

## Stretch Goals (not yet implemented)

- FAISS vector index on source doc for semantic chunk retrieval before NLI (see spec)
- FastAPI wrapper: `POST /detect` → `HallucinationReport`
- Streamlit demo UI
