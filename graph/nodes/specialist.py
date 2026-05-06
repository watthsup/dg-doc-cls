from __future__ import annotations

import time
from typing import Any
import structlog

from graph.state import GraphState
from graph.prompts import (
    MED_SPECIALIST_SYSTEM, MED_SPECIALIST_USER,
    NONMED_SPECIALIST_SYSTEM, NONMED_SPECIALIST_USER
)
from graph.logprob_analyzer import analyze_logprobs
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.multi_page import PageClassificationResult

log = structlog.get_logger()

def make_specialist_node(llm_factory: Any, is_med: bool):
    """Factory for specialist node (MED or NONMED)."""
    node_name = "med_specialist" if is_med else "nonmed_specialist"
    system_prompt = MED_SPECIALIST_SYSTEM if is_med else NONMED_SPECIALIST_SYSTEM
    user_template = MED_SPECIALIST_USER if is_med else NONMED_SPECIALIST_USER

    def specialist_node(state: GraphState) -> GraphState:
        text = state.get("azure_ocr_text", "")
        start_time = time.perf_counter()
        
        llm = llm_factory(logprobs=True)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_template.format(ocr_text=text[:10000]))
        ]
        
        response = llm.invoke(messages)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        from schemas.models import VALID_MED_SUB_CODES, VALID_NONMED_SUB_CODES
        valid_tokens = [c.value for c in (VALID_MED_SUB_CODES if is_med else VALID_NONMED_SUB_CODES)]
        
        analysis = analyze_logprobs(response.response_metadata, valid_tokens)
        sub_code = analysis.top1_token
        
        usage = response.response_metadata.get("token_usage", {})
        
        # Determine uncertainty based on margin
        is_uncertain = analysis.margin_score < 1.5 # threshold
        
        page_res = PageClassificationResult(
            page_index=state.get("page_index", 0),
            root_code=state.get("root_code", ""),
            sub_code=sub_code,
            root_margin=state.get("root_margin", 0.0),
            sub_margin=analysis.margin_score,
            root_confidence_pct=state.get("root_confidence_pct", 0.0),
            sub_confidence_pct=analysis.confidence_pct,
            root_score=state.get("root_score", 0.0),
            sub_score=analysis.top1_logprob,
            hospital_name=None, # Hospital name extraction can be added here
            is_uncertain=state.get("is_uncertain", False) or is_uncertain,
            execution_trail=state.get("execution_trail", []) + [node_name],
            ocr_text=text,
            root_logprobs=state.get("root_logprobs"),
            sub_logprobs=analysis.model_dump(),
            node_metrics={
                **state.get("node_metrics", {}),
                node_name: {
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            }
        )

        return {
            "page_results": [page_res],
            "execution_trail": [node_name],
            "node_metrics": {
                node_name: {
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            }
        }

    return specialist_node
