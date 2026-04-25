"""Adapter — converts LangGraph GraphState to existing pipeline models.

Bridges the new graph-based classification with the existing
PageResult/DocumentResult schema so all downstream consumers
(exporters, UI, batch) work without modification.
"""

from __future__ import annotations

from typing import Any

from schemas.models import (
    CODE_TO_PRIMARY,
    CODE_TO_SUBCATEGORY,
    ClassificationCode,
    ClassificationResult,
    DocumentResult,
    LogprobAnalysis,
    PageResult,
    PrimaryClass,
    ProcessingMetadata,
    QualityAssessment,
    SignalScores,
    Subcategory,
)


def graph_state_to_page_result(
    state: dict[str, Any],
    page_idx: int = 0,
    quality_assessment: QualityAssessment | None = None,
    ocr_confidence: float = 0.0,
) -> PageResult:
    """Convert a completed GraphState into a PageResult.

    Maps 3-letter codes back to the existing PrimaryClass/Subcategory enums.
    """
    root_code = state.get("root_code", "MED")
    sub_code = state.get("sub_code", "MOT")

    # Map codes to enums (with safe fallbacks)
    try:
        primary = CODE_TO_PRIMARY[ClassificationCode(root_code)]
    except (ValueError, KeyError):
        primary = PrimaryClass.MEDICAL

    try:
        subcategory = CODE_TO_SUBCATEGORY[ClassificationCode(sub_code)]
    except (ValueError, KeyError):
        subcategory = Subcategory.MEDICAL_OTHER if primary == PrimaryClass.MEDICAL else Subcategory.OTHER

    # Build quality assessment
    if quality_assessment is None:
        quality_assessment = QualityAssessment(
            issues=[],
            skew_angle=0.0,
        )

    # Compute an overall confidence from the graph's margin-based metrics
    root_conf = state.get("root_confidence_pct", 0.0) / 100.0
    sub_conf = state.get("sub_confidence_pct", 0.0) / 100.0
    overall_confidence = min(1.0, (root_conf + sub_conf) / 2.0)

    signals = SignalScores(
        ocr_confidence=ocr_confidence,
        quality_score=quality_assessment.quality_score,
    )

    return PageResult(
        page_index=page_idx,
        classification=ClassificationResult(
            primary_class=primary,
            subcategory=subcategory,
            hospital_name=state.get("hospital_name"),
        ),
        confidence=overall_confidence,
        signals=signals,
        quality_assessment=quality_assessment,
        ocr_text=state.get("azure_ocr_text", ""),
    )


def graph_state_to_document_result(
    state: dict[str, Any],
    file_name: str = "",
    filename_doc_type: str | None = None,
    quality_assessment: QualityAssessment | None = None,
    ocr_confidence: float = 0.0,
    processing_time_ms: int = 0,
    total_pages: int = 1,
) -> DocumentResult:
    """Convert a completed GraphState into a DocumentResult.

    Creates a single-page result since the graph processes the
    merged OCR text for the entire document.
    """
    page_result = graph_state_to_page_result(
        state=state,
        page_idx=0,
        quality_assessment=quality_assessment,
        ocr_confidence=ocr_confidence,
    )

    return DocumentResult(
        document_id=state.get("document_id", "unknown"),
        file_name=file_name,
        filename_doc_type=filename_doc_type,
        hospital_name=state.get("hospital_name"),
        pages=[page_result],
        processing_metadata=ProcessingMetadata(
            pages_used=[0],
            total_pages=total_pages,
            processing_time_ms=processing_time_ms,
        ),
    )


def build_logprob_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Build a human-readable summary of logprob analysis for the UI.

    Returns a dict with root and sub classification details
    including margin scores, confidence, and uncertainty flags.
    """
    return {
        "root": {
            "code": state.get("root_code"),
            "margin": state.get("root_margin", 0.0),
            "confidence_pct": state.get("root_confidence_pct", 0.0),
            "logprobs": state.get("root_logprobs"),
        },
        "sub": {
            "code": state.get("sub_code"),
            "margin": state.get("sub_margin", 0.0),
            "confidence_pct": state.get("sub_confidence_pct", 0.0),
            "logprobs": state.get("sub_logprobs"),
        },
        "is_uncertain": state.get("is_uncertain", False),
        "uncertainty_stage": state.get("uncertainty_stage"),
        "requires_human_review": state.get("requires_human_review", False),
        "execution_trail": state.get("execution_trail", []),
    }
