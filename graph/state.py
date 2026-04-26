"""LangGraph agent state definition for hierarchical classification.

The GraphState TypedDict is the single source of truth for all data
flowing through the classification pipeline. It is persisted via
the LangGraph checkpointer for async HITL resumption.
"""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    """Stateful schema for the hierarchical classification graph.

    All fields are optional (total=False) so nodes can incrementally
    populate the state as execution progresses through the pipeline.
    """

    # --- Stage 0: Document Identity & Azure Ingestion ---
    document_id: str
    file_path: str                     # Serializable string path
    file_type: str                     # "pdf" or "image"
    azure_ocr_text: str                # Full text from Azure DI
    ocr_metadata: dict[str, Any]       # Layout, Tables, Confidence
    ocr_confidence: float              # Overall OCR confidence (0-1)

    # --- Stage 1: Root Classification (MED/NON) ---
    root_code: str | None              # "MED" or "NON"
    root_logprobs: dict[str, Any] | None  # Raw logprob data from LLM
    root_margin: float                 # Margin score for root decision
    root_confidence_pct: float         # exp(logprob) * 100

    # --- Stage 2: Sub-Classification ---
    sub_code: str | None               # "LAB", "CHK", "IMG", etc.
    sub_logprobs: dict[str, Any] | None
    sub_margin: float
    sub_confidence_pct: float

    # --- Control Flow & HITL ---
    is_uncertain: bool                 # True if any margin < threshold
    requires_human_review: bool        # Breakpoint trigger status
    human_override_code: str | None    # Code selected by human reviewer
    uncertainty_stage: str | None      # "root" or "sub" — which stage is uncertain

    # --- Extracted Metadata ---
    hospital_name: str | None

    # --- Quality Data (passed through from OpenCV stage) ---
    quality_assessment: dict[str, Any] | None

    # --- Audit ---
    execution_trail: list[str]         # Ordered list of nodes visited


def create_initial_state(
    document_id: str,
    file_path: str,
    file_type: str,
) -> GraphState:
    """Create a fresh GraphState with sensible defaults."""
    return GraphState(
        document_id=document_id,
        file_path=file_path,
        file_type=file_type,
        azure_ocr_text="",
        ocr_metadata={},
        ocr_confidence=0.0,
        root_code=None,
        root_logprobs=None,
        root_margin=0.0,
        root_confidence_pct=0.0,
        sub_code=None,
        sub_logprobs=None,
        sub_margin=0.0,
        sub_confidence_pct=0.0,
        is_uncertain=False,
        requires_human_review=False,
        human_override_code=None,
        uncertainty_stage=None,
        hospital_name=None,
        quality_assessment=None,
        execution_trail=[],
    )
