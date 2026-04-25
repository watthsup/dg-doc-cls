"""LangGraph-based hierarchical classification pipeline.

Provides a stateful classification graph with logprob-based confidence
analysis and Human-in-the-Loop (HITL) breakpoints.
"""

from graph.builder import build_classification_graph

__all__ = ["build_classification_graph"]
