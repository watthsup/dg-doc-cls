from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from schemas.multi_page import PageClassificationResult


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries, keeping all keys."""
    return {**left, **right}


class GraphState(TypedDict, total=False):
    """LangGraph state schema for hierarchical classification.

    Designed for Native Map/Reduce using langgraph.types.Send.
    - Fields with Annotated[..., operator.add] aggregate results from parallel branches.
    """

    # --- Inputs ---
    document_id: Annotated[str, lambda x, y: y or x]
    file_path: Annotated[str, lambda x, y: y or x]
    file_type: Annotated[str, lambda x, y: y or x]
    azure_ocr_text: Annotated[str, lambda x, y: y or x]  # Full document text or page text

    # --- Intermediate Result (Shared across branches if needed) ---
    ocr_result: Annotated[Any, lambda x, y: y or x]  # Raw Azure DI result if doing OCR inside graph

    # --- Processing Context ---
    root_code: Annotated[str | None, lambda x, y: y or x]
    sub_code: Annotated[str | None, lambda x, y: y or x]
    is_uncertain: Annotated[bool, lambda x, y: y or x]
    requires_human_review: Annotated[bool, lambda x, y: y or x]
    uncertainty_stage: Annotated[str | None, lambda x, y: y or x]
    hospital_name: Annotated[str | None, lambda x, y: y or x]
    root_confidence_pct: Annotated[float, lambda x, y: y or x]
    sub_confidence_pct: Annotated[float, lambda x, y: y or x]
    page_index: Annotated[int, lambda x, y: y or x]
    root_margin: Annotated[float, lambda x, y: y or x]
    sub_margin: Annotated[float, lambda x, y: y or x]
    root_score: Annotated[float, lambda x, y: y or x]
    sub_score: Annotated[float, lambda x, y: y or x]
    root_logprobs: Annotated[dict | None, lambda x, y: y or x]

    # --- Results Aggregation (Map/Reduce) ---
    # Stores results from individual page extraction branches
    page_results: Annotated[list[PageClassificationResult], operator.add]

    # --- Audit & Metrics ---
    execution_trail: Annotated[list[str], operator.add]
    node_metrics: Annotated[dict[str, dict[str, Any]], merge_dicts]

    # --- Final Aggregate Output ---
    final_result: Any  # MultiPageResult
    error: str


def create_initial_state(
    document_id: str,
    file_path: str,
    file_type: str,
) -> GraphState:
    """Initialize the graph state with defaults to support both single and multi-page flows."""
    return GraphState(
        document_id=document_id,
        file_path=file_path,
        file_type=file_type,
        azure_ocr_text="",
        root_code=None,
        sub_code=None,
        is_uncertain=False,
        requires_human_review=False,
        uncertainty_stage=None,
        hospital_name=None,
        root_confidence_pct=0.0,
        sub_confidence_pct=0.0,
        page_results=[],
        execution_trail=[],
        node_metrics={},
    )
