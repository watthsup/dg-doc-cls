"""Composite confidence scoring from observable signals only.

LLM confidence is excluded — it is unreliable.
Formula: weighted sum + rule-based adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.models import SignalScores


@dataclass(frozen=True)
class ConfidenceWeights:
    """Immutable weights for the confidence formula."""

    keyword: float = 0.5
    ocr: float = 0.3
    quality: float = 0.2

    def __post_init__(self) -> None:
        total = self.keyword + self.ocr + self.quality
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Confidence weights must sum to ~1.0, got {total:.2f} "
                f"(keyword={self.keyword}, ocr={self.ocr}, quality={self.quality})"
            )


_LOW_OCR_THRESHOLD: float = 0.3
_LOW_OCR_PENALTY: float = 0.15

_LOW_QUALITY_THRESHOLD: float = 0.3
_LOW_QUALITY_PENALTY: float = 0.10

_HIGH_KEYWORD_THRESHOLD: float = 0.8
_HIGH_KEYWORD_BOOST: float = 0.10

_CONFLICT_KEYWORD_THRESHOLD: float = 0.6
_CONFLICT_OCR_THRESHOLD: float = 0.4
_CONFLICT_PENALTY: float = 0.10


def calculate_confidence(
    signals: SignalScores,
    weights: ConfidenceWeights | None = None,
) -> float:
    """Calculate composite confidence from observable signals.

    The score represents trust in the classification result
    based on input quality and keyword evidence — NOT LLM certainty.
    """
    if weights is None:
        weights = ConfidenceWeights()

    base = (
        weights.keyword * signals.keyword_score
        + weights.ocr * signals.ocr_confidence
        + weights.quality * signals.quality_score
    )

    adjustment = 0.0

    if signals.ocr_confidence < _LOW_OCR_THRESHOLD:
        adjustment -= _LOW_OCR_PENALTY

    if signals.quality_score < _LOW_QUALITY_THRESHOLD:
        adjustment -= _LOW_QUALITY_PENALTY

    if signals.keyword_score >= _HIGH_KEYWORD_THRESHOLD:
        adjustment += _HIGH_KEYWORD_BOOST

    # High keywords + low OCR = possible false positive in noisy text
    if (
        signals.keyword_score >= _CONFLICT_KEYWORD_THRESHOLD
        and signals.ocr_confidence < _CONFLICT_OCR_THRESHOLD
    ):
        adjustment -= _CONFLICT_PENALTY

    return max(0.0, min(1.0, base + adjustment))
