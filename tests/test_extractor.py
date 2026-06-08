"""Tests for the claim extractor."""

from unittest.mock import MagicMock, patch

import pytest

from hallucination_detector.extractor import ClaimExtractor
from hallucination_detector.models import Claim


SAMPLE_TEXT = "The sky is blue. Water boils at 100 degrees Celsius."
MOCK_RESPONSE = '[{"id": 1, "text": "The sky is blue.", "span_start": 0, "span_end": 15}, {"id": 2, "text": "Water boils at 100 degrees Celsius.", "span_start": 17, "span_end": 52}]'


@patch("hallucination_detector.extractor.anthropic.Anthropic")
def test_extract_returns_claims(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_RESPONSE)]
    mock_client.messages.create.return_value = mock_message

    extractor = ClaimExtractor()
    claims = extractor.extract(SAMPLE_TEXT)

    assert len(claims) == 2
    assert isinstance(claims[0], Claim)
    assert claims[0].text == "The sky is blue."
    assert claims[1].id == 2


@patch("hallucination_detector.extractor.anthropic.Anthropic")
def test_extract_strips_markdown_fences(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    fenced = f"```json\n{MOCK_RESPONSE}\n```"
    mock_message.content = [MagicMock(text=fenced)]
    mock_client.messages.create.return_value = mock_message

    extractor = ClaimExtractor()
    claims = extractor.extract(SAMPLE_TEXT)
    assert len(claims) == 2
