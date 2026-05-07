"""Document Processor — orchestrates classification via LangGraph.

Designed as a reusable module for integration into CLI scripts, APIs, 
Streamlit apps, or any service that needs document classification.

Usage:
    processor = DocumentProcessor()
    result = await processor.process_file(Path("doc.pdf"))
    result = await processor.process_text("some OCR text")
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.infrastructure.config.settings import AppConfig
from src.bootstrap import build_classification_runner
from src.domain.models.multi_page import MultiPageResult


class DocumentProcessor:
    """Reusable document classification module."""

    def __init__(self, config: AppConfig | None = None, graph_type: str = "doc_cls") -> None:
        self._config = config or AppConfig()  # type: ignore[call-arg]
        self._runner = build_classification_runner(config=self._config, graph_type=graph_type)

    async def process_file(self, file_path: Path) -> MultiPageResult:
        """Process a file: OCR followed by hierarchical classification."""
        doc_id = f"{file_path.stem}_{uuid.uuid4().hex[:8]}"
        
        return await self._runner.run(
            file_path=str(file_path),
            document_id=doc_id
        )

    async def process_text(self, text: str) -> MultiPageResult:
        """Classify pre-provided text without OCR."""
        doc_id = f"text_{uuid.uuid4().hex[:8]}"
        
        return await self._runner.run(
            ocr_text=text,
            document_id=doc_id
        )


# ----------------------------------------------------------------------
# Convenience functions (backward-compatible with scripts/notebooks)
# ----------------------------------------------------------------------

async def process_document_pages(
    file_path: Path,
    config: AppConfig | None = None,
    **kwargs: Any,
) -> MultiPageResult:
    """Convenience wrapper — creates a DocumentProcessor and processes a file."""
    processor = DocumentProcessor(config=config)
    return await processor.process_file(file_path)


async def process_text_as_page(
    text: str,
    config: AppConfig | None = None,
    **kwargs: Any,
) -> MultiPageResult:
    """Convenience wrapper — creates a DocumentProcessor and processes text."""
    processor = DocumentProcessor(config=config)
    return await processor.process_text(text)
