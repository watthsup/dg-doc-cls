from __future__ import annotations

import time
from typing import Any
import structlog

from graph.state import GraphState
from graph.prompts import MED_SPECIALIST_SYSTEM, NONMED_SPECIALIST_SYSTEM, SPECIALIST_USER
from graph.logprob_analyzer import analyze_logprobs
from langchain_core.messages import SystemMessage, HumanMessage

log = structlog.get_logger()

def make_specialist_node(llm_factory: Any, is_med: bool = True):
    """Factory for specialist nodes (Medical or Non-Medical)."""
    node_name = "med_specialist" if is_med else "nonmed_specialist"
    system_prompt = MED_SPECIALIST_SYSTEM if is_med else NONMED_SPECIALIST_SYSTEM
    
    # Define expected tokens for logprob analysis based on specialist type
    expected_tokens = ["LAB", "RX", "REC", "FIN"] if is_med else ["OTH", "N/A"]

    def specialist_node(state: GraphState) -> dict[str, Any]:
        text = state.get("azure_ocr_text", "")
        if not text:
            return {"error": f"Empty text for {node_name}"}

        start_time = time.perf_counter()
        llm = llm_factory(logprobs=True)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=SPECIALIST_USER.format(ocr_text=text[:10000]))
        ]
        
        response = llm.invoke(messages)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Analyze logprobs for the sub-category
        analysis = analyze_logprobs(response.response_metadata, expected_tokens)
        usage = response.response_metadata.get("token_usage", {})
        
        # Merge uncertainty from router if it was already uncertain
        was_uncertain = state.get("is_uncertain", False)
        is_uncertain = was_uncertain or (analysis.margin_score < 1.5)

        return {
            "sub_code": analysis.top1_token,
            "sub_margin": analysis.margin_score,
            "sub_confidence_pct": analysis.confidence_pct,
            "sub_score": analysis.top1_logprob,
            "is_uncertain": is_uncertain,
            "sub_logprobs": analysis.model_dump(),
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
