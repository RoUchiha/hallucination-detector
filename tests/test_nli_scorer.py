"""Tests for the NLI scorer."""

from unittest.mock import MagicMock, patch

import pytest

from hallucination_detector.models import Claim
from hallucination_detector.nli_scorer import NLIScorer


def make_claim(text: str, id: int = 1) -> Claim:
    return Claim(id=id, text=text, span_start=0, span_end=len(text))


@patch("hallucination_detector.nli_scorer.pipeline")
def test_entailed_claim(mock_pipeline_fn):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "labels": ["entailment", "neutral", "contradiction"],
        "scores": [0.92, 0.05, 0.03],
    }
    mock_pipeline_fn.return_value = mock_pipe

    scorer = NLIScorer()
    claim = make_claim("Paris is in France.")
    label, conf = scorer.score_claim(claim, "Paris is the capital of France.")
    assert label == "ENTAILED"
    assert conf > 0.8


@patch("hallucination_detector.nli_scorer.pipeline")
def test_contradicted_claim(mock_pipeline_fn):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "labels": ["contradiction", "neutral", "entailment"],
        "scores": [0.88, 0.08, 0.04],
    }
    mock_pipeline_fn.return_value = mock_pipe

    scorer = NLIScorer()
    claim = make_claim("The tower was built in 1800.")
    label, conf = scorer.score_claim(claim, "The tower was built in 1889.")
    assert label == "CONTRADICTED"


@patch("hallucination_detector.nli_scorer.pipeline")
def test_score_all_returns_verdicts(mock_pipeline_fn):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "labels": ["entailment", "neutral", "contradiction"],
        "scores": [0.95, 0.03, 0.02],
    }
    mock_pipeline_fn.return_value = mock_pipe

    scorer = NLIScorer()
    claims = [make_claim("Paris is in France.", i) for i in range(1, 4)]
    verdicts = scorer.score_all(claims, "Paris is located in France.")
    assert len(verdicts) == 3
    assert all(v.final_verdict == "SUPPORTED" for v in verdicts)
