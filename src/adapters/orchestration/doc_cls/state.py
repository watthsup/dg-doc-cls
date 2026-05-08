from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from src.domain.models.multi_page import PageClassificationResult, MultiPageResult


def merge_metrics(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    """Merge node metrics dictionary (latest-wins for same keys)."""
    merged = dict(existing or {})
    if incoming:
        merged.update(incoming)
    return merged


class GraphState(TypedDict, total=False):
    """
    Parent state, matching ParallelExtractState style.
    Non-annotated fields are set once.
    Annotated fields are aggregated from parallel branches.
    """
    # --- Inputs ---
    document_id: str
    file_path: str
    file_type: str
    azure_ocr_text: str
    start_time: float
    ocr_result: Any

    # --- Aggregated Results ---
    page_results: Annotated[list[PageClassificationResult], operator.add]
    node_metrics: Annotated[dict[str, Any], merge_metrics]
    
    # --- Final Output ---
    final_result: MultiPageResult
    error: str


class PageState(TypedDict):
    """Internal state for a single page branch."""
    page_index: int
    page_text: str
    document_id: str


def create_initial_state(file_path: str, document_id: str | None = None) -> GraphState:
    import time
    from pathlib import Path
    return {
        "document_id": document_id or Path(file_path).stem,
        "file_path": file_path,
        "file_type": "pdf" if file_path.lower().endswith(".pdf") else "text",
        "start_time": time.time(),
        "page_results": [],
        "node_metrics": {},
    } # type: ignore
