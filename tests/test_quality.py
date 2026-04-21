import numpy as np
import pytest
from ocr.quality import (
    assess_image_quality,
    assess_multi_page_quality,
    merge_quality,
)
from schemas.models import OCRPageResult, OCRResult, QualityAssessment


def _make_sharp_text_image() -> np.ndarray:
    """Create a sharp image with white background and black text-like blobs."""
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Add some high-contrast 'text' (boxes)
    img[20:40, 20:180] = 0
    img[60:80, 20:150] = 0
    img[100:120, 20:170] = 0
    return img


def _make_blurry_image() -> np.ndarray:
    """Create a very blurry image."""
    img = _make_sharp_text_image()
    import cv2
    return cv2.GaussianBlur(img, (21, 21), 0)


def _make_low_contrast_image() -> np.ndarray:
    """Create a low-contrast gray image."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


class TestAssessImageQuality:

    def test_sharp_image_high_blur_score(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        assert result.blur_score > 0.8

    def test_blurry_image_low_blur_score(self) -> None:
        result = assess_image_quality(_make_blurry_image())
        assert result.blur_score < 0.2

    def test_high_contrast_image(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        assert result.contrast_score > 0.8

    def test_low_contrast_image(self) -> None:
        result = assess_image_quality(_make_low_contrast_image())
        assert result.contrast_score < 0.3

    def test_quality_score_is_mean_of_signals(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        from ocr.quality import _compute_skew_score
        skew_s = _compute_skew_score(result.skew_angle)
        expected = (result.blur_score + result.contrast_score + skew_s) / 3.0
        assert abs(result.quality_score - expected) < 0.001

    def test_skew_angle_near_zero_for_straight_image(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        assert abs(result.skew_angle) < 1.0

    def test_issues_reported_for_blurry_image(self) -> None:
        result = assess_image_quality(_make_blurry_image())
        assert any("blur" in issue.lower() for issue in result.issues)

    def test_scores_in_valid_range(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        assert 0.0 <= result.quality_score <= 1.0
        assert 0.0 <= result.blur_score <= 1.0
        assert 0.0 <= result.contrast_score <= 1.0


class TestAssessMultiPageQuality:

    def test_worst_page_determines_score(self) -> None:
        sharp = _make_sharp_text_image()
        blurry = _make_blurry_image()
        result = assess_multi_page_quality([sharp, blurry])
        single_blurry = assess_image_quality(blurry)
        assert result.quality_score == pytest.approx(single_blurry.quality_score, abs=0.01)

    def test_issues_include_page_numbers(self) -> None:
        blurry = _make_blurry_image()
        result = assess_multi_page_quality([blurry, blurry])
        assert any("Page 0" in issue for issue in result.issues)
        assert any("Page 1" in issue for issue in result.issues)


class TestMergeQuality:

    def test_merge_does_not_change_score(self) -> None:
        image_q = assess_image_quality(_make_sharp_text_image())
        ocr = OCRResult(
            pages=[OCRPageResult(page_index=0, text="test", mean_confidence=95.0)],
            merged_text="test",
            overall_confidence=0.95,
        )
        merged = merge_quality(image_q, ocr)
        # Score should be EXACTLY the same as before merge
        assert merged.quality_score == image_q.quality_score

    def test_merge_concatenates_ocr_issues(self) -> None:
        image_q = assess_image_quality(_make_sharp_text_image())
        # Bad OCR confidence (e.g. 20%)
        ocr = OCRResult(
            pages=[OCRPageResult(page_index=0, text="test", mean_confidence=20.0)],
            merged_text="test",
            overall_confidence=0.20,
        )
        merged = merge_quality(image_q, ocr)
        assert any("ocr confidence" in issue.lower() for issue in merged.issues)
        assert any("Page 0" in issue for issue in merged.issues)

    def test_merge_preserves_opencv_fields(self) -> None:
        image_q = assess_image_quality(_make_sharp_text_image())
        ocr = OCRResult(pages=[], merged_text="", overall_confidence=0.0)
        merged = merge_quality(image_q, ocr)
        assert merged.blur_score == image_q.blur_score
        assert merged.contrast_score == image_q.contrast_score
        assert merged.skew_angle == image_q.skew_angle
