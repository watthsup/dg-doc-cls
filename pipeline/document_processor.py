"""Document Processor — splits multi-page files and classifies each page independently.

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
    from langgraph.graph.state import CompiledStateGraph

logger = structlog.get_logger()


def _extract_page_result(
    page_index: int,
    state: dict[str, Any],
    ocr_text: str,
) -> PageClassificationResult:
    """Convert a completed GraphState dict into a PageClassificationResult."""
    return PageClassificationResult(
        page_index=page_index,
        root_code=state.get("root_code", ""),
        sub_code=state.get("sub_code", ""),
        root_margin=state.get("root_margin", 0.0),
        sub_margin=state.get("sub_margin", 0.0),
        root_confidence_pct=state.get("root_confidence_pct", 0.0),
        sub_confidence_pct=state.get("sub_confidence_pct", 0.0),
        hospital_name=state.get("hospital_name"),
        is_uncertain=state.get("is_uncertain", False),
        execution_trail=state.get("execution_trail", []),
        ocr_text=ocr_text,
        root_logprobs=state.get("root_logprobs"),
        sub_logprobs=state.get("sub_logprobs"),
    )


async def process_document_pages(
    file_path: Path,
    config: AppConfig | None = None,
    graph: CompiledStateGraph | None = None,
    max_concurrent_pages: int = 5,
) -> MultiPageResult:
    """Process a multi-page document, classifying each page independently.

    Args:
        file_path: Path to the document file (PDF, TIF, or image).
        config: Application configuration. Created from env if None.
        graph: Pre-built LangGraph. Built fresh if None.
        max_concurrent_pages: Max pages to classify in parallel.

    Returns:
        MultiPageResult with per-page classification details.
    """
    start_time = time.monotonic()

    if config is None:
        config = AppConfig()  # type: ignore[call-arg]

    log = logger.bind(file_path=str(file_path))

    # --- 1. Validate file ---
    file_type = detect_file_type(file_path)
    if not file_type:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    doc_id = generate_document_id(file_path)
    log = log.bind(document_id=doc_id)

    # --- 2. Run Azure DI OCR on entire file (single API call) ---
    log.info("document_processor_ocr_start")
    di_client = create_di_client(config)
    ocr_result = await asyncio.to_thread(
        analyze_document,
        client=di_client,
        file_path=file_path,
        model_id=config.azure_di_model,
    )

    total_pages = len(ocr_result.pages)
    log.info(
        "document_processor_ocr_complete",
        total_pages=total_pages,
        overall_confidence=round(ocr_result.overall_confidence, 3),
    )

    if total_pages == 0:
        raise ValueError(f"Azure DI returned 0 pages for {file_path}")

    # --- 3. Build graph (once, reused for all pages) ---
    if graph is None:
        graph = build_classification_graph(config, use_checkpointer=False)

    # --- 4. Classify each page in parallel ---
    semaphore = asyncio.Semaphore(max_concurrent_pages)

    async def classify_page(page_idx: int, page_text: str) -> PageClassificationResult:
        async with semaphore:
            log.info("classifying_page", page_index=page_idx)

            # Create state with pre-populated OCR text (skips OCR node)
            state = create_initial_state(
                document_id=f"{doc_id}_p{page_idx}",
                file_path=str(file_path),
                file_type=file_type,
            )
            state["azure_ocr_text"] = page_text

            # Each page needs a unique thread_id for the graph
            thread_config = {"configurable": {"thread_id": f"{doc_id}_p{page_idx}"}}
            final_state = await graph.ainvoke(state, config=thread_config)

            return _extract_page_result(page_idx, final_state, page_text)

    # Build tasks for all pages
    tasks = []
    for ocr_page in ocr_result.pages:
        tasks.append(classify_page(ocr_page.page_index, ocr_page.text))

    page_results = await asyncio.gather(*tasks)

    # Sort by page index
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


async def process_text_as_page(
    text: str,
    config: AppConfig | None = None,
    graph: CompiledStateGraph | None = None,
) -> MultiPageResult:
    """Process raw text as a single-page document (for testing / notebook use).

    This is a convenience function that wraps text into the same
    MultiPageResult format without needing a file on disk.
    """
    start_time = time.monotonic()

    if config is None:
        config = AppConfig()  # type: ignore[call-arg]

    if graph is None:
        graph = build_classification_graph(config, use_checkpointer=False)

    state = create_initial_state(
        document_id="text_input",
        file_path="<text_input>",
        file_type="text",
    )
    state["azure_ocr_text"] = text

    thread_config = {"configurable": {"thread_id": "text_input"}}
    final_state = await graph.ainvoke(state, config=thread_config)

    page_result = _extract_page_result(0, final_state, text)
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    return MultiPageResult(
        document_id="text_input",
        file_name="<text_input>",
        total_pages=1,
        pages=[page_result],
        processing_time_ms=elapsed_ms,
    )
