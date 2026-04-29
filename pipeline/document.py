"""Document Processor — splits multi-page files and classifies each page independently.

Designed as a reusable module for integration into CLI scripts, APIs, 
Streamlit apps, or any service that needs document classification.

Usage:
    processor = DocumentProcessor(config)
    result = await processor.process_file(Path("doc.pdf"))
    result = await processor.process_text("some OCR text")

Flow:
  1. Load file and detect type (PDF / TIF / Image)
  2. Run Azure DI OCR on the entire file (single API call, page-level results)
  3. For each page: invoke the LangGraph pipeline with that page's OCR text
  4. Collect per-page results into a MultiPageResult

Design decisions:
  - Single Azure DI call for the whole file (cheaper, Azure returns page-level text)
  - Pages are classified in parallel via asyncio.gather with a configurable semaphore
  - The graph's OCR node is skipped by pre-populating azure_ocr_text in the state
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from config.settings import AppConfig
from graph.builder import build_classification_graph
from graph.state import create_initial_state
from ocr.engine import analyze_document, create_di_client
from pipeline.filesystem import detect_file_type, generate_document_id
from schemas.multi_page import MultiPageResult, PageClassificationResult

if TYPE_CHECKING:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from langgraph.graph.state import CompiledStateGraph

logger = structlog.get_logger()


class DocumentProcessor:
    """Reusable document classification module.

    Holds shared resources (config, graph, OCR client) so they are
    created once and reused across multiple calls. Thread-safe for
    concurrent async usage via semaphore control.

    Args:
        config: Application configuration. Created from env if None.
        graph: Pre-built LangGraph. Built on first use if None.
        max_concurrent_pages: Max pages to classify in parallel per document.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        graph: CompiledStateGraph | None = None,
        max_concurrent_pages: int = 5,
    ) -> None:
        self._config = config or AppConfig()  # type: ignore[call-arg]
        self._graph = graph
        self._max_concurrent_pages = max_concurrent_pages
        self._di_client: DocumentIntelligenceClient | None = None

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def graph(self) -> CompiledStateGraph:
        """Lazy-build the classification graph on first access."""
        if self._graph is None:
            self._graph = build_classification_graph(
                self._config, use_checkpointer=False,
            )
        return self._graph

    @property
    def di_client(self) -> DocumentIntelligenceClient:
        """Lazy-create the Azure DI client on first access."""
        if self._di_client is None:
            self._di_client = create_di_client(self._config)
        return self._di_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_file(self, file_path: Path) -> MultiPageResult:
        """Process a multi-page document, classifying each page independently.

        Args:
            file_path: Path to the document file (PDF, TIF, or image).

        Returns:
            MultiPageResult with per-page classification details.
        """
        start_time = time.monotonic()
        log = logger.bind(file_path=str(file_path))

        # --- 1. Validate file ---
        file_type = detect_file_type(file_path)
        if not file_type:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        doc_id = generate_document_id(file_path)
        log = log.bind(document_id=doc_id)

        # --- 2. Run Azure DI OCR on entire file (single API call) ---
        log.info("document_processor_ocr_start")
        ocr_result = await asyncio.to_thread(
            analyze_document,
            client=self.di_client,
            file_path=file_path,
            model_id=self._config.azure_di_model,
        )

        total_pages = len(ocr_result.pages)
        log.info(
            "document_processor_ocr_complete",
            total_pages=total_pages,
            overall_confidence=round(ocr_result.overall_confidence, 3),
        )

        if total_pages == 0:
            raise ValueError(f"Azure DI returned 0 pages for {file_path}")

        # --- 3. Classify each page in parallel ---
        semaphore = asyncio.Semaphore(self._max_concurrent_pages)

        async def classify_page(
            page_idx: int, page_text: str,
        ) -> PageClassificationResult:
            async with semaphore:
                log.info("classifying_page", page_index=page_idx)
                return await self._classify_text(
                    text=page_text,
                    document_id=f"{doc_id}_p{page_idx}",
                    file_path=str(file_path),
                    file_type=file_type,
                    page_index=page_idx,
                )

        tasks = [
            classify_page(p.page_index, p.text)
            for p in ocr_result.pages
        ]
        page_results = await asyncio.gather(*tasks)
        page_results = sorted(page_results, key=lambda p: p.page_index)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        log.info(
            "document_processor_complete",
            total_pages=total_pages,
            classified_pages=len(page_results),
            elapsed_ms=elapsed_ms,
        )

        return MultiPageResult(
            document_id=doc_id,
            file_name=file_path.name,
            total_pages=total_pages,
            pages=list(page_results),
            processing_time_ms=elapsed_ms,
        )

    async def process_text(self, text: str) -> MultiPageResult:
        """Process raw text as a single-page document.

        Useful for testing, notebooks, or when OCR text is already available.
        """
        start_time = time.monotonic()

        page_result = await self._classify_text(
            text=text,
            document_id="text_input",
            file_path="<text_input>",
            file_type="text",
            page_index=0,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return MultiPageResult(
            document_id="text_input",
            file_name="<text_input>",
            total_pages=1,
            pages=[page_result],
            processing_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _classify_text(
        self,
        text: str,
        document_id: str,
        file_path: str,
        file_type: str,
        page_index: int,
    ) -> PageClassificationResult:
        """Run a single page's text through the LangGraph classification pipeline."""
        state = create_initial_state(
            document_id=document_id,
            file_path=file_path,
            file_type=file_type,
        )
        state["azure_ocr_text"] = text

        thread_config = {"configurable": {"thread_id": document_id}}
        final_state = await self.graph.ainvoke(state, config=thread_config)

        return _extract_page_result(page_index, final_state, text)


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------


def _extract_page_result(
    page_index: int,
    state: dict[str, Any],
    ocr_text: str,
) -> PageClassificationResult:
    """Convert a completed GraphState dict into a PageClassificationResult."""
    root_lp = state.get("root_logprobs")
    sub_lp = state.get("sub_logprobs")

    return PageClassificationResult(
        page_index=page_index,
        root_code=state.get("root_code", ""),
        sub_code=state.get("sub_code", ""),
        root_margin=state.get("root_margin", 0.0),
        sub_margin=state.get("sub_margin", 0.0),
        root_confidence_pct=state.get("root_confidence_pct", 0.0),
        sub_confidence_pct=state.get("sub_confidence_pct", 0.0),
        root_score=root_lp.get("top1_logprob", 0.0) if root_lp else 0.0,
        sub_score=sub_lp.get("top1_logprob", 0.0) if sub_lp else 0.0,
        hospital_name=state.get("hospital_name"),
        is_uncertain=state.get("is_uncertain", False),
        execution_trail=state.get("execution_trail", []),
        ocr_text=ocr_text,
        root_logprobs=root_lp,
        sub_logprobs=sub_lp,
    )


# ----------------------------------------------------------------------
# Convenience functions (backward-compatible with scripts/notebooks)
# ----------------------------------------------------------------------


async def process_document_pages(
    file_path: Path,
    config: AppConfig | None = None,
    graph: CompiledStateGraph | None = None,
    max_concurrent_pages: int = 5,
) -> MultiPageResult:
    """Convenience wrapper — creates a DocumentProcessor and processes a file."""
    processor = DocumentProcessor(
        config=config, graph=graph, max_concurrent_pages=max_concurrent_pages,
    )
    return await processor.process_file(file_path)


async def process_text_as_page(
    text: str,
    config: AppConfig | None = None,
    graph: CompiledStateGraph | None = None,
) -> MultiPageResult:
    """Convenience wrapper — creates a DocumentProcessor and processes text."""
    processor = DocumentProcessor(config=config, graph=graph)
    return await processor.process_text(text)
