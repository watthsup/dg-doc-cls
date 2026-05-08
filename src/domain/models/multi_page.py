from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

class PageClassificationResult(BaseModel):
    page_index: int = Field(description="0-indexed page number")
    root_code: str = Field(description="Root classification (MED / NON)")
    sub_code: str = Field(description="Sub-classification (LAB, CHK, ID, etc.)")
    
    root_margin: float = Field(default=0.0, description="Logprob margin at root level")
    sub_margin: float = Field(default=0.0, description="Logprob margin at specialist level")
    root_confidence_pct: float = Field(default=0.0, description="Root confidence percentage (0-100)")
    sub_confidence_pct: float = Field(default=0.0, description="Specialist confidence percentage (0-100)")
    root_score: float = Field(default=0.0, description="Raw logprob score of winning root token")
    sub_score: float = Field(default=0.0, description="Raw logprob score of winning sub-code token")
    
    is_uncertain: bool = Field(default=False)
    execution_trail: list[str] = Field(default_factory=list)
    ocr_text: str = Field(default="", description="OCR text for this page (for debugging)")

    root_logprobs: dict | None = Field(default=None)
    sub_logprobs: dict | None = Field(default=None)
    
    node_metrics: dict[str, dict[str, Any]] | None = Field(default=None)


class MultiPageResult(BaseModel):
    document_id: str
    file_name: str
    total_pages: int
    pages: list[PageClassificationResult] = Field(default_factory=list)
    processing_time_ms: int = Field(default=0)
    pipeline_metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_uncertain_pages(self) -> bool:
        return any(p.is_uncertain for p in self.pages)

    @property
    def summary(self) -> str:
        parts = [f"P{p.page_index + 1}:{p.root_code}->{p.sub_code}" for p in self.pages]
        return " | ".join(parts)
