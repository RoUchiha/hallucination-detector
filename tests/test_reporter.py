"""Tests for the report generator."""

import json
import tempfile
from pathlib import Path

from hallucination_detector.models import Claim, ClaimVerdict
from hallucination_detector.reporter import build_report, save_json


def _make_verdict(id: int, final: str) -> ClaimVerdict:
    return ClaimVerdict(
        claim=Claim(id=id, text=f"Claim {id}", span_start=0, span_end=10),
        nli_label="ENTAILED" if final == "SUPPORTED" else "CONTRADICTED",
        nli_confidence=0.9,
        final_verdict=final,
    )


def test_hallucination_rate():
    verdicts = [
        _make_verdict(1, "SUPPORTED"),
        _make_verdict(2, "HALLUCINATED"),
        _make_verdict(3, "HALLUCINATED"),
        _make_verdict(4, "SUPPORTED"),
    ]
    report = build_report(verdicts, "source", "generated")
    assert report.total_claims == 4
    assert report.hallucination_rate == 0.5
    assert report.risk_tier == "MEDIUM"


def test_risk_tier_low():
    verdicts = [_make_verdict(i, "SUPPORTED") for i in range(1, 6)]
    report = build_report(verdicts, "s", "g")
    assert report.risk_tier == "LOW"


def test_risk_tier_high():
    verdicts = [_make_verdict(i, "HALLUCINATED") for i in range(1, 4)]
    report = build_report(verdicts, "s", "g")
    assert report.risk_tier == "HIGH"


def test_save_json_roundtrip():
    verdicts = [_make_verdict(1, "SUPPORTED")]
    report = build_report(verdicts, "src", "gen")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    save_json(report, path)
    data = json.loads(Path(path).read_text())
    assert data["total_claims"] == 1
    assert data["risk_tier"] == "LOW"
