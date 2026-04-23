"""All Pydantic schemas for the classification system.

These models define the contracts between pipeline stages.
LLMOutput is the structured output from the LLM call — it gets
merged with computed signals to form the final DocumentResult.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums — closed label space
# ---------------------------------------------------------------------------


class PrimaryClass(StrEnum):
    """Level-1 classification: broad document category."""

    MEDICAL = "medical"
    NON_MEDICAL = "non_medical"


class Subcategory(StrEnum):
    """Level-2 classification: specific document type."""

    LAB = "lab"
    HEALTH_CHECK = "health_check"
    IMAGING_REPORT = "imaging_report"
    ENCOUNTER_RECORDS = "encounter_records"
    MEDICAL_CERTIFICATE = "medical_certificate"
    DISCHARGE_SUMMARY = "discharge_summary"
    MEDICAL_OTHER = "medical_other"
    ID = "id"
    FINANCIAL = "financial"
    OTHER = "other"


VALID_SUBCATEGORIES: dict[PrimaryClass, set[Subcategory]] = {
    PrimaryClass.MEDICAL: {
        Subcategory.LAB,
        Subcategory.HEALTH_CHECK,
        Subcategory.IMAGING_REPORT,
        Subcategory.ENCOUNTER_RECORDS,
        Subcategory.PRESCRIPTIONS,
        Subcategory.MEDICAL_CERTIFICATE,
        Subcategory.DISCHARGE_SUMMARY,
        Subcategory.MEDICAL_OTHER,
    },
    PrimaryClass.NON_MEDICAL: {
        Subcategory.ID,
        Subcategory.FINANCIAL,
        Subcategory.OTHER,
    },
}


# ---------------------------------------------------------------------------
# LLM structured output — single call returns this
# ---------------------------------------------------------------------------


class LLMOutput(BaseModel):
    """Structured output from a single LLM call.

    Contains classification AND hospital name extraction in one shot.
    This is the Pydantic model passed to `with_structured_output()`.
    The LLM returns valid JSON matching this schema — no text parsing needed.
    """

    primary_class: PrimaryClass
    subcategory: Subcategory
    hospital_name: str | None = Field(
        default=None,
        description="Hospital or clinic name if this is a medical document, null otherwise",
    )


# ---------------------------------------------------------------------------
# Pipeline stage models
# ---------------------------------------------------------------------------


class DocumentInput(BaseModel):
    """Input to the pipeline — a single document to classify."""

    document_id: str = Field(description="Unique identifier for this document")
    file_path: Path = Field(description="Absolute path to the document file")
    file_type: Literal["pdf", "image"] = Field(description="Detected file type")


class OCRWordResult(BaseModel):
    """A single word extracted by OCR with its confidence."""

    text: str
    confidence: float = Field(
        ge=0.0, le=100.0,
        description="OCR confidence 0-100 (Azure DI returns 0-1, scaled to 0-100)",
    )


class OCRPageResult(BaseModel):
    """OCR output for a single page."""

    page_index: int = Field(ge=0)
    text: str = Field(description="Full extracted text for this page")
    words: list[OCRWordResult] = Field(default_factory=list)
    mean_confidence: float = Field(
        ge=0.0, le=100.0,
        description="Average word confidence (0-100 scale)",
    )


class OCRResult(BaseModel):
    """Merged OCR output across all processed pages."""

    pages: list[OCRPageResult] = Field(default_factory=list)
    merged_text: str = Field(default="")
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Normalized average confidence (0-1 scale)",
    )


class ClassificationResult(BaseModel):
    """Classification portion of the final output."""

    primary_class: PrimaryClass
    subcategory: Subcategory
    hospital_name: str | None = Field(
        default=None,
        description="Extracted hospital name (only for medical documents)",
    )


class SignalScores(BaseModel):
    """Observable signals used for confidence calculation.

    LLM confidence is explicitly excluded — it is unreliable.
    """

    ocr_confidence: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)


class QualityAssessment(BaseModel):
    """Combined document quality from OpenCV image metrics.

    OCR confidence is tracked separately in SignalScores to keep signals orthogonal.
    """

    issues: list[str] = Field(default_factory=list)
    skew_angle: float = Field(description="Detected skew angle in degrees")
    # --- Internal scores (excluded from JSON output) ---
    blur_score: float = Field(default=0.0, ge=0.0, le=1.0, exclude=True)
    contrast_score: float = Field(default=0.0, ge=0.0, le=1.0, exclude=True)
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        exclude=True,
        description="Combined quality: 0.6 * image_quality + 0.4 * ocr_quality",
    )


class ProcessingMetadata(BaseModel):
    """Technical details about the execution."""

    pages_used: list[int] = Field(default_factory=list)
    total_pages: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Final output — what gets written to JSONL
# ---------------------------------------------------------------------------


class PageResult(BaseModel):
    """Classification result for a single page."""

    page_index: int = Field(ge=0)
    classification: ClassificationResult
    confidence: float = Field(ge=0.0, le=1.0)
    signals: SignalScores
    quality_assessment: QualityAssessment
    ocr_text: str = Field(
        default="",
        description="Text extracted by OCR for this page",
    )


class DocumentResult(BaseModel):
    """Final output for a single document, containing per-page results."""

    document_id: str
    file_name: str = Field(
        description="Original filename of the processed document",
    )
    filename_doc_type: str | None = Field(
        default=None,
        description="Document type extracted from filename pattern (3rd part)",
    )
    pages: list[PageResult] = Field(
        default_factory=list,
        description="Detailed classification results for each page",
    )
    processing_metadata: ProcessingMetadata = Field(default_factory=ProcessingMetadata)


class DocumentError(BaseModel):
    """Captures a processing failure for a single document."""

    document_id: str
    error_type: str
    error_message: str
    stage: str = Field(description="Pipeline stage where the error occurred")


class BatchResult(BaseModel):
    """Aggregated result from processing a batch of documents."""

    total_documents: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[DocumentResult] = Field(default_factory=list)
    errors: list[DocumentError] = Field(default_factory=list)
