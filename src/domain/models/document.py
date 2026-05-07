from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.domain.models.classification import PrimaryClass, Subcategory

class DocumentInput(BaseModel):
    document_id: str = Field(description="Unique identifier for this document")
    file_path: Path = Field(description="Absolute path to the document file")
    file_type: Literal["pdf", "image"] = Field(description="Detected file type")

class OCRWordResult(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=100.0)

class OCRPageResult(BaseModel):
    page_index: int = Field(ge=0)
    text: str = Field(description="Full extracted text for this page")
    words: list[OCRWordResult] = Field(default_factory=list)
    mean_confidence: float = Field(ge=0.0, le=100.0)

class OCRResult(BaseModel):
    pages: list[OCRPageResult] = Field(default_factory=list)
    merged_text: str = Field(default="")
    overall_confidence: float = Field(ge=0.0, le=1.0)

class ClassificationResult(BaseModel):
    primary_class: PrimaryClass
    subcategory: Subcategory
    hospital_name: str | None = Field(default=None)

class SignalScores(BaseModel):
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)

class QualityAssessment(BaseModel):
    issues: list[str] = Field(default_factory=list)
    skew_angle: float = Field(description="Detected skew angle in degrees")
    blur_score: float = Field(default=0.0, ge=0.0, le=1.0, exclude=True)
    contrast_score: float = Field(default=0.0, ge=0.0, le=1.0, exclude=True)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, exclude=True)

class ProcessingMetadata(BaseModel):
    pages_used: list[int] = Field(default_factory=list)
    total_pages: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(default=0, ge=0)

class PageResult(BaseModel):
    page_index: int = Field(ge=0)
    classification: ClassificationResult
    confidence: float = Field(ge=0.0, le=1.0)
    signals: SignalScores
    quality_assessment: QualityAssessment
    ocr_text: str = Field(default="")

class DocumentResult(BaseModel):
    document_id: str
    file_name: str
    filename_doc_type: str | None = Field(default=None)
    hospital_name: str | None = Field(default=None)
    pages: list[PageResult] = Field(default_factory=list)
    processing_metadata: ProcessingMetadata = Field(default_factory=ProcessingMetadata)

class DocumentError(BaseModel):
    document_id: str
    error_type: str
    error_message: str
    stage: str

class BatchResult(BaseModel):
    total_documents: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[DocumentResult] = Field(default_factory=list)
    errors: list[DocumentError] = Field(default_factory=list)
