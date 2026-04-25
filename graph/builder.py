"""Graph builder — constructs and compiles the classification LangGraph.

The graph implements a hierarchical decision flow:
  OCR → Root Router → (Med Specialist | NonMed Specialist) → [HITL Gateway]

Routing is driven by logprob margin analysis at each classification node.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from graph.nodes import (
    hitl_gateway_node,
    med_specialist_node,
    nonmed_specialist_node,
    ocr_ingestion_node,
    root_router_node,
)
from graph.state import GraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from config.settings import AppConfig


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_root(state: GraphState) -> Literal["med_specialist", "nonmed_specialist", "hitl_gateway"]:
    """Route based on root classification and margin confidence."""
    if state.get("is_uncertain", False):
        return "hitl_gateway"

    root_code = state.get("root_code", "MED")
    if root_code == "MED":
        return "med_specialist"
    return "nonmed_specialist"


def route_after_specialist(state: GraphState) -> Literal["hitl_gateway", "__end__"]:
    """Route based on specialist margin confidence."""
    if state.get("is_uncertain", False) and state.get("uncertainty_stage") == "sub":
        return "hitl_gateway"
    return END


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_classification_graph(config: AppConfig) -> CompiledStateGraph:
    """Build and compile the hierarchical classification LangGraph.

    Args:
        config: Application configuration with margin threshold
                and checkpoint DB path.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    builder = StateGraph(GraphState)

    # --- Register nodes ---
    builder.add_node("ocr_ingestion", ocr_ingestion_node)
    builder.add_node("root_router", root_router_node)
    builder.add_node("med_specialist", med_specialist_node)
    builder.add_node("nonmed_specialist", nonmed_specialist_node)
    builder.add_node("hitl_gateway", hitl_gateway_node)

    # --- Define edges ---
    builder.set_entry_point("ocr_ingestion")
    builder.add_edge("ocr_ingestion", "root_router")

    builder.add_conditional_edges(
        "root_router",
        route_after_root,
        {
            "med_specialist": "med_specialist",
            "nonmed_specialist": "nonmed_specialist",
            "hitl_gateway": "hitl_gateway",
        },
    )

    builder.add_conditional_edges(
        "med_specialist",
        route_after_specialist,
        {
            "hitl_gateway": "hitl_gateway",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "nonmed_specialist",
        route_after_specialist,
        {
            "hitl_gateway": "hitl_gateway",
            END: END,
        },
    )

    # HITL gateway always ends (state is persisted for human review)
    builder.add_edge("hitl_gateway", END)

    # --- Compile with checkpointer ---
    checkpoint_path = Path(config.checkpoint_db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpointer = SqliteSaver.from_conn_string(str(checkpoint_path))

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_gateway"],
    )
