from __future__ import annotations

from typing import Protocol
from src.domain.models.multi_page import PageClassificationResult

class PageClassifier(Protocol):
    def classify_page(self, page_index: int, text: str) -> tuple[PageClassificationResult, dict]:
        """Classify a single page and return the result + metrics."""
        ...
