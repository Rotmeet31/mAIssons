"""
Tests for analysis.py
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestParseResponse:
    def test_valid_json_parsed_correctly(self):
        from analysis import _parse_response
        raw = json.dumps({
            "bias_label": "left",
            "bias_score": -0.6,
            "confidence": 0.85,
            "framing_summary": "Uses progressive framing.",
        })
        result = _parse_response(raw)
        assert result["bias_label"] == "left"
        assert result["bias_score"] == pytest.approx(-0.6)
        assert result["confidence"] == pytest.approx(0.85)

    def test_strips_markdown_fences(self):
        from analysis import _parse_response
        raw = "```json\n{\"bias_label\":\"center\",\"bias_score\":0.1,\"confidence\":0.9,\"framing_summary\":\"Neutral.\"}\n```"
        result = _parse_response(raw)
        assert result["bias_label"] == "center"

    def test_raises_on_invalid_bias_label(self):
        from analysis import _parse_response
        raw = json.dumps({
            "bias_label": "extreme",
            "bias_score": 0.5,
            "confidence": 0.7,
            "framing_summary": "Some summary.",
        })
        with pytest.raises(ValueError, match="bias_label"):
            _parse_response(raw)

    def test_raises_on_missing_field(self):
        from analysis import _parse_response
        raw = json.dumps({
            "bias_label": "right",
            "bias_score": 0.8,
            # confidence and framing_summary missing
        })
        with pytest.raises(ValueError, match="Missing fields"):
            _parse_response(raw)

    def test_raises_on_out_of_range_bias_score(self):
        from analysis import _parse_response
        raw = json.dumps({
            "bias_label": "right",
            "bias_score": 2.5,
            "confidence": 0.9,
            "framing_summary": "Out of range.",
        })
        with pytest.raises(ValueError, match="bias_score out of range"):
            _parse_response(raw)

    def test_raises_on_invalid_json(self):
        from analysis import _parse_response
        with pytest.raises(json.JSONDecodeError):
            _parse_response("not json at all")


class TestAnalyzeArticle:
    def _make_mock_client(self, raw_response: str):
        usage = MagicMock()
        usage.prompt_token_count = 100
        usage.candidates_token_count = 50

        response = MagicMock()
        response.text = raw_response
        response.usage_metadata = usage

        client = MagicMock()
        client.models.generate_content.return_value = response
        return client

    def test_successful_analysis(self):
        from analysis import analyze_article
        raw = json.dumps({
            "bias_label": "right",
            "bias_score": 0.7,
            "confidence": 0.9,
            "framing_summary": "Uses conservative framing.",
        })
        client = self._make_mock_client(raw)
        result = analyze_article(client, 1, "Test title", "Test body", "Fox News")
        assert result is not None
        assert result["bias_label"] == "right"
        assert result["prompt_tokens"] == 100
        assert result["response_tokens"] == 50

    def test_retries_on_parse_failure_then_succeeds(self):
        from analysis import analyze_article
        good = json.dumps({
            "bias_label": "center",
            "bias_score": 0.0,
            "confidence": 0.8,
            "framing_summary": "Balanced reporting.",
        })
        usage = MagicMock()
        usage.prompt_token_count = 50
        usage.candidates_token_count = 20

        bad_resp = MagicMock()
        bad_resp.text = "invalid json"
        bad_resp.usage_metadata = usage

        good_resp = MagicMock()
        good_resp.text = good
        good_resp.usage_metadata = usage

        client = MagicMock()
        client.models.generate_content.side_effect = [bad_resp, good_resp]

        result = analyze_article(client, 2, "Title", "Body", "Reuters")
        assert result is not None
        assert result["bias_label"] == "center"
        assert client.models.generate_content.call_count == 2

    def test_returns_none_on_api_error(self):
        from analysis import analyze_article
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("API down")
        result = analyze_article(client, 3, "T", "B", "S")
        assert result is None
