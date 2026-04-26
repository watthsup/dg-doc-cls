"""Tests for graph/logprob_analyzer.py — the Reliability Engine."""

from __future__ import annotations

import math

import pytest

from graph.logprob_analyzer import analyze_logprobs


class TestAnalyzeLogprobs:
    """Test the margin calculation engine."""

    def test_high_confidence_margin(self) -> None:
        """When Top1 dominates, margin should be large."""
        metadata = {
            "logprobs": {
                "content": [
                    {
                        "token": "MED",
                        "logprob": -0.01,
                        "top_logprobs": [
                            {"token": "MED", "logprob": -0.01},
                            {"token": "NON", "logprob": -4.5},
                        ],
                    }
                ]
            }
        }
        result = analyze_logprobs(metadata, ["MED", "NON"])
        assert result.top1_token == "MED"
        assert result.top2_token == "NON"
        assert result.margin_score == pytest.approx(-0.01 - (-4.5))
        assert result.margin_score > 1.5  # High confidence

    def test_low_confidence_margin(self) -> None:
        """When Top1 and Top2 are close, margin should be small."""
        metadata = {
            "logprobs": {
                "content": [
                    {
                        "token": "MED",
                        "logprob": -0.5,
                        "top_logprobs": [
                            {"token": "MED", "logprob": -0.5},
                            {"token": "NON", "logprob": -0.8},
                        ],
                    }
                ]
            }
        }
        result = analyze_logprobs(metadata, ["MED", "NON"])
        assert result.margin_score == pytest.approx(0.3)
        assert result.margin_score < 1.5  # Low confidence — should trigger HITL

    def test_confidence_pct_calculation(self) -> None:
        """exp(logprob) * 100 should give probability percentage."""
        metadata = {
            "logprobs": {
                "content": [
                    {
                        "token": "LAB",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "LAB", "logprob": -0.1},
                            {"token": "CHK", "logprob": -3.0},
                        ],
                    }
                ]
            }
        }
        result = analyze_logprobs(metadata, ["LAB", "CHK", "IMG", "MOT"])
        expected_pct = math.exp(-0.1) * 100
        assert result.confidence_pct == pytest.approx(expected_pct, abs=0.1)

    def test_filters_invalid_tokens(self) -> None:
        """Only valid classification tokens should be considered."""
        metadata = {
            "logprobs": {
                "content": [
                    {
                        "token": "MED",
                        "logprob": -0.01,
                        "top_logprobs": [
                            {"token": "MED", "logprob": -0.01},
                            {"token": "The", "logprob": -0.02},
                            {"token": "NON", "logprob": -5.0},
                        ],
                    }
                ]
            }
        }
        result = analyze_logprobs(metadata, ["MED", "NON"])
        # "The" should be filtered out; Top2 should be "NON", not "The"
        assert result.top1_token == "MED"
        assert result.top2_token == "NON"

    def test_no_logprobs_returns_zero_confidence(self) -> None:
        """When no logprobs are available, return safe defaults."""
        result = analyze_logprobs({}, ["MED", "NON"])
        assert result.top1_token == "UNK"
        assert result.margin_score == 0.0
        assert result.confidence_pct == 0.0

    def test_empty_content_logprobs(self) -> None:
        """When content logprobs list is empty."""
        metadata = {"logprobs": {"content": []}}
        result = analyze_logprobs(metadata, ["MED", "NON"])
        assert result.margin_score == 0.0

    def test_single_valid_token_max_confidence(self) -> None:
        """When only one valid token appears, margin should be very large."""
        metadata = {
            "logprobs": {
                "content": [
                    {
                        "token": "FIN",
                        "logprob": -0.05,
                        "top_logprobs": [
                            {"token": "FIN", "logprob": -0.05},
                            {"token": "random", "logprob": -0.1},
                        ],
                    }
                ]
            }
        }
        result = analyze_logprobs(metadata, ["ID_", "FIN", "OTH"])
        assert result.top1_token == "FIN"
        assert result.top2_token == "N/A"  # No other valid token
        # Margin should be very large (top1 - (-100))
        assert result.margin_score > 50.0

    def test_case_insensitive_matching(self) -> None:
        """Token matching should be case-insensitive."""
        metadata = {
            "logprobs": {
                "content": [
                    {
                        "token": "med",
                        "logprob": -0.01,
                        "top_logprobs": [
                            {"token": "med", "logprob": -0.01},
                            {"token": "non", "logprob": -3.0},
                        ],
                    }
                ]
            }
        }
        result = analyze_logprobs(metadata, ["MED", "NON"])
        assert result.top1_token == "MED"
        assert result.top2_token == "NON"
