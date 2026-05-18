"""
Tests for analysis.py — _parse_and_validate and analyze_cluster.
"""
import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _valid_cluster_dict(**overrides) -> dict:
    base = {
        "summary": "Something happened. Multiple outlets covered it. Details follow.",
        "shared_ground": ["Both sides report X (Reuters, BBC)", "All agree on Y (AP, Fox)"],
        "left_not_right": [{"claim": "Left emphasises civilian casualties.", "coverage": "omitted"}],
        "right_not_left": [{"claim": "Right focuses on border security.", "coverage": "downplayed"}],
        "center_angle": "Reuters added international reaction that neither side covered.",
    }
    base.update(overrides)
    return base


class TestParseAndValidate:
    def test_valid_json_returns_model(self):
        from analysis import _parse_and_validate
        raw = json.dumps(_valid_cluster_dict())
        result = _parse_and_validate(raw)
        assert result.summary.startswith("Something happened")
        assert len(result.shared_ground) == 2

    def test_strips_markdown_fences(self):
        from analysis import _parse_and_validate
        raw = "```json\n" + json.dumps(_valid_cluster_dict()) + "\n```"
        result = _parse_and_validate(raw)
        assert result.summary.startswith("Something happened")

    def test_strips_think_tags(self):
        from analysis import _parse_and_validate
        inner = json.dumps(_valid_cluster_dict())
        raw = f"<think>model reasoning here</think>{inner}"
        result = _parse_and_validate(raw)
        assert len(result.shared_ground) == 2

    def test_coverage_items_parsed_correctly(self):
        from analysis import _parse_and_validate
        result = _parse_and_validate(json.dumps(_valid_cluster_dict()))
        assert result.left_not_right[0].coverage == "omitted"
        assert result.right_not_left[0].coverage == "downplayed"

    def test_coerces_string_shared_ground_to_list(self):
        from analysis import _parse_and_validate
        data = _valid_cluster_dict(shared_ground="single string point")
        result = _parse_and_validate(json.dumps(data))
        assert isinstance(result.shared_ground, list)
        assert result.shared_ground == ["single string point"]

    def test_coerces_plain_string_coverage_items(self):
        from analysis import _parse_and_validate
        data = _valid_cluster_dict(left_not_right=["just a plain string"])
        result = _parse_and_validate(json.dumps(data))
        assert result.left_not_right[0].claim == "just a plain string"
        assert result.left_not_right[0].coverage == "downplayed"

    def test_raises_on_invalid_json(self):
        from analysis import _parse_and_validate
        with pytest.raises(json.JSONDecodeError):
            _parse_and_validate("not valid json")

    def test_raises_on_empty_shared_ground(self):
        from analysis import _parse_and_validate
        data = _valid_cluster_dict(shared_ground=[])
        with pytest.raises(ValidationError):
            _parse_and_validate(json.dumps(data))

    def test_raises_on_missing_summary(self):
        from analysis import _parse_and_validate
        data = {"shared_ground": ["Point 1"]}
        with pytest.raises((ValidationError, KeyError)):
            _parse_and_validate(json.dumps(data))


class TestAnalyzeCluster:
    def _make_openai_response(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens

        msg = MagicMock()
        msg.content = content

        choice = MagicMock()
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    def test_successful_analysis_returns_dict(self):
        from analysis import analyze_cluster
        raw = json.dumps(_valid_cluster_dict())
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response(raw)

        result = analyze_cluster(client, cluster_id=1, headline="Test headline", articles=[
            {"source_lean": "left", "source_name": "Guardian", "title": "T1", "body": "Body1"},
            {"source_lean": "right", "source_name": "Fox", "title": "T2", "body": "Body2"},
        ])
        assert result is not None
        assert isinstance(result["shared_ground"], list)
        assert isinstance(result["left_not_right"], list)
        assert isinstance(result["right_not_left"], list)
        assert "summary" in result
        assert "center_angle" in result

    def test_retries_on_validation_error_then_succeeds(self):
        from analysis import analyze_cluster
        bad_raw = json.dumps({"shared_ground": [], "summary": ""})  # fails validation
        good_raw = json.dumps(_valid_cluster_dict())

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            self._make_openai_response(bad_raw),
            self._make_openai_response(good_raw),
        ]

        result = analyze_cluster(client, cluster_id=2, headline="H", articles=[
            {"source_lean": "left", "source_name": "G", "title": "T", "body": "B"},
        ])
        assert result is not None
        assert client.chat.completions.create.call_count == 2

        second_call_messages = client.chat.completions.create.call_args_list[1][1]["messages"]
        assert any("validation" in str(m).lower() or "error" in str(m).lower()
                   for m in second_call_messages)

    def test_returns_none_after_max_retries(self):
        from analysis import analyze_cluster
        bad_raw = "not json at all"
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response(bad_raw)

        result = analyze_cluster(client, cluster_id=3, headline="H", articles=[
            {"source_lean": "center", "source_name": "BBC", "title": "T", "body": "B"},
        ])
        assert result is None

    def test_raises_on_api_error(self):
        from analysis import analyze_cluster
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        with pytest.raises(RuntimeError, match="API down"):
            analyze_cluster(client, cluster_id=4, headline="H", articles=[
                {"source_lean": "left", "source_name": "G", "title": "T", "body": "B"},
            ])
