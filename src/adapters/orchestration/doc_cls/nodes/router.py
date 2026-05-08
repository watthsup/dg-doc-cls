from __future__ import annotations

import time
from typing import Any
import structlog

from src.adapters.orchestration.doc_cls.state import GraphState
from src.application.prompts.classifier_prompts import ROOT_ROUTER_SYSTEM, ROOT_ROUTER_USER
from src.domain.services.logprob_analyzer import analyze_logprobs
from langchain_core.messages import SystemMessage, HumanMessage

log = structlog.get_logger()

def make_root_router_node(llm_factory: Any):
    """Factory for root router node."""
    def root_router_node(state: GraphState) -> dict[str, Any]:
        text = state.get("azure_ocr_text", "")
        if not text:
            return {"error": "Empty text for router"}

        start_time = time.perf_counter()
        # Enable logprobs for uncertainty detection
        llm = llm_factory(logprobs=True)
        
        messages = [
            SystemMessage(content=ROOT_ROUTER_SYSTEM),
            HumanMessage(content=ROOT_ROUTER_USER.format(ocr_text=text[:10000]))
        ]
        
        response = llm.invoke(messages)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Analyze logprobs
        analysis = analyze_logprobs(response.response_metadata, ["MED", "NON"])
        usage = response.response_metadata.get("token_usage", {})
        
        # Update branch state
        return {
            "root_code": analysis.top1_token,
            "root_margin": analysis.margin_score,
            "root_confidence_pct": analysis.confidence_pct,
            "root_score": analysis.top1_logprob,
            "is_uncertain": analysis.margin_score < 1.5,
            "root_logprobs": analysis.model_dump(),
            "execution_trail": ["root_router"],
            "node_metrics": {
                "root_router": {
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            }
        }

    return root_router_node
