from __future__ import annotations

import time
from typing import Any
import structlog

from src.adapters.orchestration.doc_cls.state import PageState, GraphState
from src.application.prompts.classifier_prompts import (
    ROOT_ROUTER_SYSTEM,
    ROOT_ROUTER_USER,
    MED_SPECIALIST_SYSTEM,
    NONMED_SPECIALIST_SYSTEM,
    MED_SPECIALIST_USER,
    NONMED_SPECIALIST_USER
)
from src.domain.services.logprob_analyzer import analyze_logprobs
from src.application.ports.input.page_classifier import PageClassifier
from src.domain.models.multi_page import PageClassificationResult
from langchain_core.messages import SystemMessage, HumanMessage

log = structlog.get_logger()

def make_classify_page_node(classifier: Any):
    """
    Factory for the unified page classification node.
    Follows the style of extract_page node in doc-structure-agent.
    """
    def classify_page_node(state: PageState) -> dict[str, Any]:
        page_index = state["page_index"]
        text = state["page_text"]
        
        try:
            result, metrics = classifier.classify_page(page_index, text)
            return {
                "page_results": [result],
                "node_metrics": metrics
            }
        except Exception as e:
            log.error("page_classification_failed", page_index=page_index, error=str(e))
            # Follow doc-structure-agent: return empty list on failure to let other pages proceed
            return {"page_results": []}

    return classify_page_node
