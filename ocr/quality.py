"""Document quality assessment — two complementary signal sources.

1. OpenCV (pixel-level) — runs BEFORE Azure DI on locally rendered images:
   - blur_score: Laplacian variance (sharp images have high variance)
   - contrast_score: grayscale std dev
   - skew_angle: contour minAreaRect analysis

2. OCR confidence (text-level) — runs AFTER Azure DI returns results:
   - ocr_quality_score: worst-page mean word confidence

3. merge_quality() — combines both into a single QualityAssessment:
   - quality_score = 0.6 * image_quality + 0.4 * ocr_quality
   - issues from both sources are concatenated
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from schemas.models import OCRResult, QualityAssessment

if TYPE_CHECKING:
    from numpy.typing import NDArray

# --- Weights for combining image and OCR quality signals ---
_IMAGE_QUALITY_WEIGHT: float = 0.6
_OCR_QUALITY_WEIGHT: float = 0.4

# --- OpenCV thresholds ---
_BLUR_ISSUE_THRESHOLD: float = 0.3
_SKEW_ISSUE_THRESHOLD: float = 5.0
_CONTRAST_ISSUE_THRESHOLD: float = 0.3

# --- OCR confidence thresholds ---
_LOW_CONFIDENCE_THRESHOLD: float = 0.5
_VERY_LOW_CONFIDENCE_THRESHOLD: float = 0.3
_MIN_WORDS_PER_PAGE: int = 5


# ---------------------------------------------------------------------------
# OpenCV quality (pixel-level)
# ---------------------------------------------------------------------------


def _compute_blur_score(gray: NDArray[np.uint8], threshold: float) -> float:
    """Blur via Variance of Laplacian — sharp images have high variance."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance: float = float(laplacian.var())
    return float(min(1.0, variance / threshold))


def _compute_skew_angle(gray: NDArray[np.uint8]) -> float:
    """Estimate document skew via contour minAreaRect analysis."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    angles: list[float] = []
    min_contour_area = gray.shape[0] * gray.shape[1] * 0.001
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45:
            angle += 90
        angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def _compute_contrast_score(gray: NDArray[np.uint8], min_std: float) -> float:
    """Contrast from grayscale intensity std dev."""
    std_dev: float = float(np.std(gray))
    return float(min(1.0, std_dev / (min_std * 3.0)))


def _compute_skew_score(angle: float) -> float:
    """Normalize skew angle (0-20 degrees) into 0.0-1.0 score."""
    # 0 deg = 1.0 score, 20+ deg = 0.0 score
    return float(max(0.0, 1.0 - (abs(angle) / 20.0)))


def assess_image_quality(
    image: NDArray[np.uint8],
    blur_threshold: float = 100.0,
    contrast_min: float = 30.0,
) -> QualityAssessment:
    """Run OpenCV quality checks on a single page image.

    Returns a partial QualityAssessment — ocr_quality_score is 0.0
    and quality_score is image-only. Call merge_quality() after OCR
    to get the full combined assessment.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    blur_score = _compute_blur_score(gray, blur_threshold)
    skew_angle = _compute_skew_angle(gray)
    skew_score = _compute_skew_score(skew_angle)
    contrast_score = _compute_contrast_score(gray, contrast_min)

    issues: list[str] = []
    if blur_score < _BLUR_ISSUE_THRESHOLD:
        issues.append(f"High blur detected (score: {blur_score:.2f})")
    if abs(skew_angle) > _SKEW_ISSUE_THRESHOLD:
        issues.append(f"Significant skew detected ({skew_angle:.1f}°)")
    if contrast_score < _CONTRAST_ISSUE_THRESHOLD:
        issues.append(f"Low contrast detected (score: {contrast_score:.2f})")

    # Image quality is now average of Blur, Contrast, and Skew
    image_quality = (blur_score + contrast_score + skew_score) / 3.0

    return QualityAssessment(
        blur_score=blur_score,
        skew_angle=skew_angle,
        contrast_score=contrast_score,
        ocr_quality_score=0.0,  # Not yet available — set by merge_quality()
        issues=issues,
        quality_score=image_quality,  # Image-only until merged
    )


def assess_multi_page_quality(
    images: list[NDArray[np.uint8]],
    blur_threshold: float = 100.0,
    contrast_min: float = 30.0,
) -> QualityAssessment:
    """Assess OpenCV quality across pages — worst page determines score."""
    if not images:
        return QualityAssessment(
            blur_score=0.0,
            skew_angle=0.0,
            contrast_score=0.0,
            ocr_quality_score=0.0,
            issues=["No images provided for quality assessment"],
            quality_score=0.0,
        )

    assessments = [
        assess_image_quality(img, blur_threshold, contrast_min) for img in images
    ]
    worst = min(assessments, key=lambda a: a.quality_score)

    all_issues: list[str] = []
    for i, a in enumerate(assessments):
        for issue in a.issues:
            all_issues.append(f"Page {i}: {issue}")

    return QualityAssessment(
        blur_score=worst.blur_score,
        skew_angle=worst.skew_angle,
        contrast_score=worst.contrast_score,
        ocr_quality_score=0.0,  # Set by merge_quality() after OCR
        issues=all_issues,
        quality_score=worst.quality_score,
    )


# ---------------------------------------------------------------------------
# OCR confidence quality (text-level)
# ---------------------------------------------------------------------------


def _ocr_quality_score_from_result(ocr_result: OCRResult) -> tuple[float, list[str]]:
    """Extract OCR quality score and issues from Azure DI result.

    Returns (worst_page_score 0–1, issues list).
    """
    if not ocr_result.pages:
        return 0.0, ["No pages available for OCR quality assessment"]

    issues: list[str] = []
    page_scores: list[float] = []

    for page in ocr_result.pages:
        page_conf = page.mean_confidence / 100.0
        page_scores.append(page_conf)
        label = f"Page {page.page_index}"

        if len(page.words) < _MIN_WORDS_PER_PAGE:
            issues.append(
                f"{label}: Very few words detected ({len(page.words)})"
                " — possibly blank or image-only"
            )
        if page_conf < _VERY_LOW_CONFIDENCE_THRESHOLD:
            issues.append(
                f"{label}: Very low OCR confidence ({page_conf:.2f})"
                " — likely poor quality or handwritten"
            )
        elif page_conf < _LOW_CONFIDENCE_THRESHOLD:
            issues.append(
                f"{label}: Low OCR confidence ({page_conf:.2f})"
                " — may affect classification accuracy"
            )

    return min(page_scores), issues


# ---------------------------------------------------------------------------
# Merge: combine OpenCV + OCR into final QualityAssessment
# ---------------------------------------------------------------------------


def merge_quality(
    image_quality: QualityAssessment,
    ocr_result: OCRResult,
) -> QualityAssessment:
    """Merge OpenCV quality with OCR confidence into a single assessment.

    quality_score = 0.6 * image_quality + 0.4 * ocr_quality

    The two signals are complementary:
    - Image quality catches visual problems (blur, skew) even before OCR
    - OCR quality catches readability problems (handwriting, heavy noise)
      that pixel-level metrics may miss
    """
    ocr_score, ocr_issues = _ocr_quality_score_from_result(ocr_result)

    combined_score = (
        _IMAGE_QUALITY_WEIGHT * image_quality.quality_score
        + _OCR_QUALITY_WEIGHT * ocr_score
    )

    return QualityAssessment(
        blur_score=image_quality.blur_score,
        skew_angle=image_quality.skew_angle,
        contrast_score=image_quality.contrast_score,
        ocr_quality_score=ocr_score,
        issues=image_quality.issues + ocr_issues,
        quality_score=min(1.0, max(0.0, combined_score)),
    )
