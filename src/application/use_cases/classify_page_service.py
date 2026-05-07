from __future__ import annotations

import time
import structlog
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from src.application.ports.input.page_classifier import PageClassifier
from src.domain.models.multi_page import PageClassificationResult
from src.application.prompts.classifier_prompts import (
    ROOT_ROUTER_SYSTEM, ROOT_ROUTER_USER,
    MED_SPECIALIST_SYSTEM, NONMED_SPECIALIST_SYSTEM, 
    MED_SPECIALIST_USER, NONMED_SPECIALIST_USER
)
from src.domain.services.logprob_analyzer import analyze_logprobs
from src.domain.models.classification import (
    VALID_ROOT_CODES,
    VALID_MED_SUB_CODES,
    VALID_NONMED_SUB_CODES,
)

log = structlog.get_logger()

class ClassifyPageService:
    def __init__(self, llm_factory: Any):
        # We are keeping llm_factory for now, ideally it uses LLMClient
        self.llm_factory = llm_factory

    def classify_page(self, page_index: int, text: str) -> tuple[PageClassificationResult, dict]:
        node_metrics = {}
        execution_trail = []
        
        # 1. Root Routing
        router_start = time.perf_counter()
        llm_router = self.llm_factory(logprobs=True)
        router_messages = [
            SystemMessage(content=ROOT_ROUTER_SYSTEM),
            HumanMessage(content=ROOT_ROUTER_USER.format(ocr_text=text[:10000]))
        ]
        router_resp = llm_router.invoke(router_messages)
        router_latency = int((time.perf_counter() - router_start) * 1000)
        
        # Analyze Router results
        root_analysis = analyze_logprobs(router_resp.response_metadata, [c.value for c in VALID_ROOT_CODES])
        usage_r = router_resp.response_metadata.get("token_usage", {})
        node_metrics["root_router"] = {
            "latency_ms": router_latency,
            "prompt_tokens": usage_r.get("prompt_tokens", 0),
            "completion_tokens": usage_r.get("completion_tokens", 0),
            "total_tokens": usage_r.get("total_tokens", 0),
        }
        execution_trail.append("root_router")
        
        root_code = root_analysis.top1_token
        is_uncertain = root_analysis.margin_score < 1.5
        
        # 2. Specialist Classification
        sub_code = None
        sub_analysis = None
        if root_code == "MED":
            spec_start = time.perf_counter()
            llm_spec = self.llm_factory(logprobs=True)
            spec_messages = [
                SystemMessage(content=MED_SPECIALIST_SYSTEM),
                HumanMessage(content=MED_SPECIALIST_USER.format(ocr_text=text[:10000]))
            ]
            spec_resp = llm_spec.invoke(spec_messages)
            spec_latency = int((time.perf_counter() - spec_start) * 1000)
            
            sub_analysis = analyze_logprobs(spec_resp.response_metadata, [c.value for c in VALID_MED_SUB_CODES])
            usage_s = spec_resp.response_metadata.get("token_usage", {})
            node_metrics["med_specialist"] = {
                "latency_ms": spec_latency,
                "prompt_tokens": usage_s.get("prompt_tokens", 0),
                "completion_tokens": usage_s.get("completion_tokens", 0),
                "total_tokens": usage_s.get("total_tokens", 0),
            }
            execution_trail.append("med_specialist")
            sub_code = sub_analysis.top1_token
            is_uncertain = is_uncertain or (sub_analysis.margin_score < 1.5)
        else:
            spec_start = time.perf_counter()
            llm_spec = self.llm_factory(logprobs=True)
            spec_messages = [
                SystemMessage(content=NONMED_SPECIALIST_SYSTEM),
                HumanMessage(content=NONMED_SPECIALIST_USER.format(ocr_text=text[:10000]))
            ]
            spec_resp = llm_spec.invoke(spec_messages)
            spec_latency = int((time.perf_counter() - spec_start) * 1000)
            
            sub_analysis = analyze_logprobs(spec_resp.response_metadata, [c.value for c in VALID_NONMED_SUB_CODES])
            usage_s = spec_resp.response_metadata.get("token_usage", {})
            node_metrics["nonmed_specialist"] = {
                "latency_ms": spec_latency,
                "prompt_tokens": usage_s.get("prompt_tokens", 0),
                "completion_tokens": usage_s.get("completion_tokens", 0),
                "total_tokens": usage_s.get("total_tokens", 0),
            }
            execution_trail.append("nonmed_specialist")
            sub_code = sub_analysis.top1_token
            is_uncertain = is_uncertain or (sub_analysis.margin_score < 1.5)

        result = PageClassificationResult(
            page_index=page_index,
            root_code=root_code,
            sub_code=sub_code,
            root_margin=root_analysis.margin_score,
            sub_margin=sub_analysis.margin_score if sub_analysis else 0.0,
            root_confidence_pct=root_analysis.confidence_pct,
            sub_confidence_pct=sub_analysis.confidence_pct if sub_analysis else 0.0,
            root_score=root_analysis.top1_logprob,
            sub_score=sub_analysis.top1_logprob if sub_analysis else 0.0,
            is_uncertain=is_uncertain,
            execution_trail=execution_trail,
            ocr_text=text,
            root_logprobs=root_analysis.model_dump(),
            sub_logprobs=sub_analysis.model_dump() if sub_analysis else None,
            node_metrics=node_metrics
        )
        
        return result, node_metrics
