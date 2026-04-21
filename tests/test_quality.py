"""Tests for quality assessor — OpenCV pixel-level + OCR confidence merge."""

import numpy as np
import pytest

from ocr.quality import assess_image_quality, assess_multi_page_quality, merge_quality
from schemas.models import OCRPageResult, OCRResult, OCRWordResult


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------


def _make_sharp_text_image(height: int = 400, width: int = 600) -> np.ndarray:
    image = np.ones((height, width), dtype=np.uint8) * 255
    for y in range(50, height - 50, 30):
        image[y : y + 3, 50 : width - 50] = 0
    return image


def _make_blurry_image(height: int = 400, width: int = 600) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(_make_sharp_text_image(height, width), (51, 51), 0)


def _make_low_contrast_image(height: int = 400, width: int = 600) -> np.ndarray:
    return np.random.randint(120, 130, (height, width), dtype=np.uint8)


# ---------------------------------------------------------------------------
# OCR result helpers
# ---------------------------------------------------------------------------


def _make_ocr_result(page_confidences: list[list[float]]) -> OCRResult:
    """Build OCRResult from per-page word confidences (0–1 scale)."""
    pages: list[OCRPageResult] = []
    for i, confs in enumerate(page_confidences):
        words = [
            OCRWordResult(text=f"word{j}", confidence=c * 100)
            for j, c in enumerate(confs)
        ]
        mean_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
        pages.append(
            OCRPageResult(
                page_index=i,
                text=" ".join(w.text for w in words),
                words=words,
                mean_confidence=mean_conf,
            )
        )
    overall = sum(p.mean_confidence for p in pages) / len(pages) / 100.0 if pages else 0.0
    return OCRResult(
        pages=pages,
        merged_text="\n\n".join(p.text for p in pages),
        overall_confidence=min(1.0, max(0.0, overall)),
    )


# ---------------------------------------------------------------------------
# OpenCV quality tests
# ---------------------------------------------------------------------------


class TestAssessImageQuality:

    def test_sharp_image_high_blur_score(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        assert result.blur_score > 0.5

    def test_blurry_image_low_blur_score(self) -> None:
        result = assess_image_quality(_make_blurry_image())
        assert result.blur_score < 0.3

    def test_high_contrast_image(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        assert result.contrast_score > 0.5

    def test_low_contrast_image(self) -> None:
        result = assess_image_quality(_make_low_contrast_image())
        assert result.contrast_score < 0.3

    def test_quality_score_is_mean_of_blur_and_contrast(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        expected = (result.blur_score + result.contrast_score) / 2.0
        assert abs(result.quality_score - expected) < 0.001

    def test_skew_angle_near_zero_for_straight_image(self) -> None:
        result = assess_image_quality(_make_sharp_text_image())
        assert abs(result.skew_angle) < 5.0

    def test_issues_reported_for_blurry_image(self) -> None:
        result = assess_image_quality(_make_blurry_image())
        assert any("blur" in issue.lower() for issue in result.issues)

    def test_issues_reported_for_low_contrast(self) -> None:
        result = assess_image_quality(_make_low_contrast_image())
        assert any("contrast" in issue.lower() for issue in result.issues)

    def test_scores_in_valid_range(self) -> None:
        for make_image in [
            _make_sharp_text_image,
            _make_blurry_image,
            _make_low_contrast_image,
        ]:
            result = assess_image_quality(make_image())
            assert 0.0 <= result.blur_score <= 1.0
            assert 0.0 <= result.contrast_score <= 1.0
            assert 0.0 <= result.quality_score <= 1.0

    def test_bgr_image_input(self) -> None:
        gray = _make_sharp_text_image()
        bgr = np.stack([gray, gray, gray], axis=-1)
        result = assess_image_quality(bgr)
        assert result.quality_score > 0.0

    def test_ocr_quality_score_zero_before_merge(self) -> None:
        """Before merge, ocr_quality_score is 0.0 (placeholder)."""
        result = assess_image_quality(_make_sharp_text_image())
        assert result.ocr_quality_score == 0.0


class TestAssessMultiPageQuality:

    def test_worst_page_determines_score(self) -> None:
        sharp = _make_sharp_text_image()
        blurry = _make_blurry_image()
        result = assess_multi_page_quality([sharp, blurry])
        single_blurry = assess_image_quality(blurry)
        assert result.quality_score == pytest.approx(single_blurry.quality_score, abs=0.01)

    def test_empty_images_list(self) -> None:
        result = assess_multi_page_quality([])
        assert result.quality_score == 0.0
        assert len(result.issues) > 0

    def test_issues_include_page_numbers(self) -> None:
        blurry = _make_blurry_image()
        result = assess_multi_page_quality([blurry, blurry])
        assert any("Page 0" in issue for issue in result.issues)
        assert any("Page 1" in issue for issue in result.issues)

    def test_single_page_list(self) -> None:
        image = _make_sharp_text_image()
        multi_result = assess_multi_page_quality([image])
        single_result = assess_image_quality(image)
        assert multi_result.quality_score == pytest.approx(
            single_result.quality_score, abs=0.01
        )


# ---------------------------------------------------------------------------
# merge_quality tests
# ---------------------------------------------------------------------------


class TestMergeQuality:

    def test_merge_combines_scores(self) -> None:
        """Combined score = 0.6 * image + 0.4 * ocr."""
        image_q = assess_image_quality(_make_sharp_text_image())
        ocr = _make_ocr_result([[0.8, 0.85, 0.9, 0.75, 0.82]])
        merged = merge_quality(image_q, ocr)

        expected = 0.6 * image_q.quality_score + 0.4 * merged.ocr_quality_score
        assert merged.quality_score == pytest.approx(expected, abs=0.01)

    def test_merge_sets_ocr_quality_score(self) -> None:
        """After merge, ocr_quality_score should be non-zero for good OCR."""
        image_q = assess_image_quality(_make_sharp_text_image())
        ocr = _make_ocr_result([[0.9, 0.88, 0.92, 0.87, 0.91]])
        merged = merge_quality(image_q, ocr)
        assert merged.ocr_quality_score > 0.5

    def test_merge_preserves_opencv_fields(self) -> None:
        """blur_score, contrast_score, skew_angle come from OpenCV result."""
        image_q = assess_image_quality(_make_sharp_text_image())
        ocr = _make_ocr_result([[0.9, 0.88, 0.92, 0.87, 0.91]])
        merged = merge_quality(image_q, ocr)

        assert merged.blur_score == pytest.approx(image_q.blur_score, abs=0.001)
        assert merged.contrast_score == pytest.approx(image_q.contrast_score, abs=0.001)
        assert merged.skew_angle == pytest.approx(image_q.skew_angle, abs=0.1)

    def test_merge_concatenates_issues(self) -> None:
        """Issues from both OpenCV and OCR are combined."""
        blurry_image = assess_image_quality(_make_blurry_image())
        # Low confidence OCR to trigger OCR issues
        ocr = _make_ocr_result([[0.2, 0.15, 0.18, 0.22, 0.19]])
        merged = merge_quality(blurry_image, ocr)

        has_blur_issue = any("blur" in i.lower() for i in merged.issues)
        has_ocr_issue = any("confidence" in i.lower() for i in merged.issues)
        assert has_blur_issue
        assert has_ocr_issue

    def test_merge_quality_in_valid_range(self) -> None:
        """Combined quality score always between 0 and 1."""
        for make_image in [_make_sharp_text_image, _make_blurry_image]:
            for confs in [[[0.95, 0.9, 0.88, 0.92, 0.87]], [[0.1, 0.15, 0.2, 0.12, 0.18]]]:
                image_q = assess_image_quality(make_image())
                ocr = _make_ocr_result(confs)
                merged = merge_quality(image_q, ocr)
                assert 0.0 <= merged.quality_score <= 1.0

    def test_merge_empty_ocr(self) -> None:
        """Merging with empty OCR result should still return valid assessment."""
        image_q = assess_image_quality(_make_sharp_text_image())
        empty_ocr = OCRResult(pages=[], merged_text="", overall_confidence=0.0)
        merged = merge_quality(image_q, empty_ocr)
        assert 0.0 <= merged.quality_score <= 1.0
        assert any(issue for issue in merged.issues)  # OCR issues should be added

    def test_high_image_low_ocr_gives_medium_quality(self) -> None:
        """Good image + bad OCR = middle ground quality."""
        image_q = assess_image_quality(_make_sharp_text_image())
        bad_ocr = _make_ocr_result([[0.1, 0.12, 0.15, 0.11, 0.13]])
        merged = merge_quality(image_q, bad_ocr)
        # Not great, not terrible — somewhere in between
        assert 0.2 < merged.quality_score < 0.9
