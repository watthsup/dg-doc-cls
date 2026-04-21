"""Tests for confidence calculator."""

import pytest

from confidence.calculator import ConfidenceWeights, calculate_confidence
from schemas.models import SignalScores


class TestConfidenceWeights:

    def test_default_weights_sum_to_one(self) -> None:
        w = ConfidenceWeights()
        assert w.keyword + w.ocr + w.quality == pytest.approx(1.0)

    def test_custom_valid_weights(self) -> None:
        w = ConfidenceWeights(keyword=0.4, ocr=0.4, quality=0.2)
        assert w.keyword == 0.4

    def test_invalid_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="must sum to ~1.0"):
            ConfidenceWeights(keyword=0.5, ocr=0.5, quality=0.5)

    def test_weights_immutable(self) -> None:
        w = ConfidenceWeights()
        with pytest.raises(AttributeError):
            w.keyword = 0.9  # type: ignore[misc]


class TestCalculateConfidence:

    def test_perfect_signals(self) -> None:
        signals = SignalScores(ocr_confidence=1.0, keyword_score=1.0, quality_score=1.0)
        assert calculate_confidence(signals) >= 0.9

    def test_zero_signals(self) -> None:
        signals = SignalScores(ocr_confidence=0.0, keyword_score=0.0, quality_score=0.0)
        assert calculate_confidence(signals) == 0.0

    def test_medium_signals(self) -> None:
        signals = SignalScores(ocr_confidence=0.5, keyword_score=0.5, quality_score=0.5)
        confidence = calculate_confidence(signals)
        assert 0.3 <= confidence <= 0.7

    def test_low_ocr_penalized(self) -> None:
        high_ocr = SignalScores(ocr_confidence=0.8, keyword_score=0.5, quality_score=0.8)
        low_ocr = SignalScores(ocr_confidence=0.2, keyword_score=0.5, quality_score=0.8)
        assert calculate_confidence(high_ocr) > calculate_confidence(low_ocr)

    def test_low_quality_penalized(self) -> None:
        high_q = SignalScores(ocr_confidence=0.8, keyword_score=0.5, quality_score=0.8)
        low_q = SignalScores(ocr_confidence=0.8, keyword_score=0.5, quality_score=0.1)
        assert calculate_confidence(high_q) > calculate_confidence(low_q)

    def test_high_keyword_boosted(self) -> None:
        low_kw = SignalScores(ocr_confidence=0.7, keyword_score=0.3, quality_score=0.7)
        high_kw = SignalScores(ocr_confidence=0.7, keyword_score=0.9, quality_score=0.7)
        assert calculate_confidence(high_kw) > calculate_confidence(low_kw)

    def test_signal_conflict_penalized(self) -> None:
        no_conflict = SignalScores(ocr_confidence=0.7, keyword_score=0.7, quality_score=0.7)
        conflict = SignalScores(ocr_confidence=0.2, keyword_score=0.7, quality_score=0.7)
        assert calculate_confidence(no_conflict) > calculate_confidence(conflict)

    def test_always_in_valid_range(self) -> None:
        test_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        for ocr in test_values:
            for kw in test_values:
                for qual in test_values:
                    signals = SignalScores(
                        ocr_confidence=ocr, keyword_score=kw, quality_score=qual
                    )
                    confidence = calculate_confidence(signals)
                    assert 0.0 <= confidence <= 1.0

    def test_custom_weights(self) -> None:
        signals = SignalScores(ocr_confidence=1.0, keyword_score=0.0, quality_score=0.0)
        weights = ConfidenceWeights(keyword=0.0, ocr=1.0, quality=0.0)
        assert calculate_confidence(signals, weights=weights) >= 0.8

    def test_deterministic(self) -> None:
        signals = SignalScores(ocr_confidence=0.6, keyword_score=0.4, quality_score=0.7)
        results = [calculate_confidence(signals) for _ in range(10)]
        assert len(set(results)) == 1
