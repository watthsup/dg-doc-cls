"""LangGraph node functions for the hierarchical classification pipeline.

Each node receives and returns a partial GraphState dict.

Design decisions:
- All LLM calls are async (ainvoke) to avoid blocking the event loop.
- ocr_ingestion_node wraps the synchronous Azure DI SDK call in
  asyncio.to_thread to prevent stalling the event loop during I/O.
- LLM clients are created inside each node call — shared clients across
  async tasks cause connection pool contention in the Azure SDK.
- Config is loaded once per node to avoid re-reading env on every call.
  A shared config object should be injected via RunnableConfig in future.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from config.settings import AppConfig
from graph.logprob_analyzer import analyze_logprobs
from graph.prompts import (
    HOSPITAL_EXTRACTION_SYSTEM,
    HOSPITAL_EXTRACTION_USER,
    MED_SPECIALIST_SYSTEM,
    MED_SPECIALIST_USER,
    NONMED_SPECIALIST_SYSTEM,
    NONMED_SPECIALIST_USER,
    ROOT_ROUTER_SYSTEM,
    ROOT_ROUTER_USER,
)
from graph.state import GraphState
from ocr.engine import analyze_document, create_di_client
from schemas.models import VALID_MED_SUB_CODES, VALID_NONMED_SUB_CODES, VALID_ROOT_CODES

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _load_config() -> AppConfig:
    """Load application configuration.

    Technical Debt: Config is re-instantiated per graph execution.
    Long-term, inject AppConfig via LangGraph RunnableConfig so it
    flows through the graph without repeated env reads.
    """
    return AppConfig()  # type: ignore[call-arg]


def _make_token_provider() -> Any:
    """Create a reusable Azure AD token provider.

    DefaultAzureCredential walks the credential chain only once
    per process — subsequent calls are cheap cache hits.
    """
    return get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )


def _create_llm(config: AppConfig, *, logprobs: bool = False) -> AzureChatOpenAI:
    """Instantiate an AzureChatOpenAI client.

    Args:
        config: Application configuration.
        logprobs: If True, enables logprob capture (incompatible with
                  with_structured_output / tool-call mode).

    Why per-call instantiation: The Azure SDK connection pool is NOT
    thread-safe when shared across concurrent asyncio tasks.
    Each node invocation gets an isolated connection pool.
    """
    token_provider = _make_token_provider()
    model_kwargs: dict[str, Any] = {}
    if logprobs:
        model_kwargs = {
            "logprobs": True,
            "top_logprobs": config.logprobs_top_n,
        }
    return AzureChatOpenAI(
        azure_deployment=config.azure_openai_deployment,
        azure_endpoint=config.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=config.azure_openai_api_version,
        temperature=0.0,
        timeout=config.llm_timeout,
        model_kwargs=model_kwargs,
    )


# ---------------------------------------------------------------------------
# Node 0: Azure OCR Ingestion
# ---------------------------------------------------------------------------


async def ocr_ingestion_node(state: GraphState) -> dict[str, Any]:
    """Call Azure Document Intelligence and populate OCR fields in state.

    The Azure DI SDK is synchronous — we wrap it in asyncio.to_thread
    to avoid blocking the event loop while awaiting the remote API call.
    This allows other concurrent graph executions to proceed.
    """
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_ocr_ingestion_start")

    config = _load_config()
    file_path = Path(state["file_path"])

    # Offload synchronous blocking SDK call to a thread pool worker
    ocr_result = await asyncio.to_thread(
        analyze_document,
        client=create_di_client(config),
        file_path=file_path,
        model_id=config.azure_di_model,
    )

    log.info(
        "graph_ocr_ingestion_complete",
        text_length=len(ocr_result.merged_text),
        ocr_confidence=round(ocr_result.overall_confidence, 3),
        pages=len(ocr_result.pages),
    )

    return {
        "azure_ocr_text": ocr_result.merged_text,
        "ocr_metadata": {
            "pages": len(ocr_result.pages),
            "overall_confidence": ocr_result.overall_confidence,
        },
        "ocr_confidence": ocr_result.overall_confidence,
        "execution_trail": state.get("execution_trail", []) + ["ocr_ingestion"],
    }


# ---------------------------------------------------------------------------
# Node 1: Root Router (MED / NON)
# ---------------------------------------------------------------------------


async def root_router_node(state: GraphState) -> dict[str, Any]:
    """Classify document into MED or NON using single-token logprob analysis.

    Uses a raw LLM call (not with_structured_output) to preserve
    response_metadata containing the logprob distribution.
    The margin between Top1 and Top2 drives the confidence decision.
    """
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_root_router_start")

    config = _load_config()
    llm = _create_llm(config, logprobs=True)
    ocr_text = state.get("azure_ocr_text", "")
    valid_tokens = [c.value for c in VALID_ROOT_CODES]

    messages = [
        SystemMessage(content=ROOT_ROUTER_SYSTEM),
        HumanMessage(content=ROOT_ROUTER_USER.format(ocr_text=ocr_text[:4000])),
    ]

    response = await llm.ainvoke(messages)
    raw_code = response.content.strip().upper()  # type: ignore[union-attr]

    analysis = analyze_logprobs(response.response_metadata, valid_tokens)

    # Prefer the generated content token; fall back to Top1 logprob token
    root_code = raw_code if raw_code in valid_tokens else analysis.top1_token
    if root_code not in valid_tokens:
        root_code = "MED"  # Final safe fallback

    is_uncertain = analysis.margin_score < config.margin_threshold

    log.info(
        "graph_root_router_complete",
        root_code=root_code,
        margin=round(analysis.margin_score, 3),
        confidence_pct=round(analysis.confidence_pct, 1),
        is_uncertain=is_uncertain,
    )

    return {
        "root_code": root_code,
        "root_logprobs": {
            "top1_token": analysis.top1_token,
            "top1_logprob": analysis.top1_logprob,
            "top2_token": analysis.top2_token,
            "top2_logprob": analysis.top2_logprob,
        },
        "root_margin": analysis.margin_score,
        "root_confidence_pct": analysis.confidence_pct,
        "is_uncertain": is_uncertain,
        "uncertainty_stage": "root" if is_uncertain else None,
        "execution_trail": state.get("execution_trail", []) + ["root_router"],
    }


# ---------------------------------------------------------------------------
# Node 2a: Medical Specialist
# ---------------------------------------------------------------------------


async def med_specialist_node(state: GraphState) -> dict[str, Any]:
    """Sub-classify a medical document into its specific document type.

    Runs two async operations concurrently:
    - Sub-classification LLM call (logprob-enabled)
    - Hospital name extraction (separate plain LLM call)

    This parallelism saves ~50% of latency vs sequential calls.
    """
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_med_specialist_start")

    config = _load_config()
    ocr_text = state.get("azure_ocr_text", "")
    valid_tokens = [c.value for c in VALID_MED_SUB_CODES]

    messages = [
        SystemMessage(content=MED_SPECIALIST_SYSTEM),
        HumanMessage(content=MED_SPECIALIST_USER.format(ocr_text=ocr_text[:4000])),
    ]

    # Run classification and hospital extraction concurrently
    llm = _create_llm(config, logprobs=True)
    response, hospital_name = await asyncio.gather(
        llm.ainvoke(messages),
        _extract_hospital_name(ocr_text, config),
    )

    raw_code = response.content.strip().upper()  # type: ignore[union-attr]
    analysis = analyze_logprobs(response.response_metadata, valid_tokens)

    sub_code = raw_code if raw_code in valid_tokens else analysis.top1_token
    if sub_code not in valid_tokens:
        sub_code = "OTH"  # Explicit fallback to shared OTH

    is_uncertain = (analysis.margin_score < config.margin_threshold) or (sub_code == "OTH")

    log.info(
        "graph_med_specialist_complete",
        sub_code=sub_code,
        margin=round(analysis.margin_score, 3),
        confidence_pct=round(analysis.confidence_pct, 1),
        is_uncertain=is_uncertain,
        hospital_name=hospital_name,
    )

    return {
        "sub_code": sub_code,
        "sub_logprobs": {
            "top1_token": analysis.top1_token,
            "top1_logprob": analysis.top1_logprob,
            "top2_token": analysis.top2_token,
            "top2_logprob": analysis.top2_logprob,
        },
        "sub_margin": analysis.margin_score,
        "sub_confidence_pct": analysis.confidence_pct,
        # Preserve root-level uncertainty — once uncertain, stays uncertain
        "is_uncertain": is_uncertain or state.get("is_uncertain", False),
        "uncertainty_stage": "sub" if is_uncertain else state.get("uncertainty_stage"),
        "hospital_name": hospital_name,
        "execution_trail": state.get("execution_trail", []) + ["med_specialist"],
    }


# ---------------------------------------------------------------------------
# Node 2b: Non-Medical Specialist
# ---------------------------------------------------------------------------


async def nonmed_specialist_node(state: GraphState) -> dict[str, Any]:
    """Sub-classify a non-medical document into its specific document type."""
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_nonmed_specialist_start")

    config = _load_config()
    ocr_text = state.get("azure_ocr_text", "")
    valid_tokens = [c.value for c in VALID_NONMED_SUB_CODES]

    messages = [
        SystemMessage(content=NONMED_SPECIALIST_SYSTEM),
        HumanMessage(content=NONMED_SPECIALIST_USER.format(ocr_text=ocr_text[:4000])),
    ]

    llm = _create_llm(config, logprobs=True)
    response = await llm.ainvoke(messages)

    raw_code = response.content.strip().upper()  # type: ignore[union-attr]
    analysis = analyze_logprobs(response.response_metadata, valid_tokens)

    sub_code = raw_code if raw_code in valid_tokens else analysis.top1_token
    if sub_code not in valid_tokens:
        sub_code = "OTH"  # Explicit fallback to shared OTH

    is_uncertain = (analysis.margin_score < config.margin_threshold) or (sub_code == "OTH")

    log.info(
        "graph_nonmed_specialist_complete",
        sub_code=sub_code,
        margin=round(analysis.margin_score, 3),
        confidence_pct=round(analysis.confidence_pct, 1),
        is_uncertain=is_uncertain,
    )

    return {
        "sub_code": sub_code,
        "sub_logprobs": {
            "top1_token": analysis.top1_token,
            "top1_logprob": analysis.top1_logprob,
            "top2_token": analysis.top2_token,
            "top2_logprob": analysis.top2_logprob,
        },
        "sub_margin": analysis.margin_score,
        "sub_confidence_pct": analysis.confidence_pct,
        "is_uncertain": is_uncertain or state.get("is_uncertain", False),
        "uncertainty_stage": "sub" if is_uncertain else state.get("uncertainty_stage"),
        "hospital_name": None,  # Non-medical docs don't have a hospital name
        "execution_trail": state.get("execution_trail", []) + ["nonmed_specialist"],
    }


# ---------------------------------------------------------------------------
# Node 3: HITL Gateway
# ---------------------------------------------------------------------------


def hitl_gateway_node(state: GraphState) -> dict[str, Any]:
    """Mark state for human review before the pipeline is interrupted.

    This node runs synchronously and only mutates state flags.
    The actual execution breakpoint is enforced by the LangGraph
    `interrupt_before=["hitl_gateway"]` compile-time directive.

    After human review, the graph is resumed via graph.invoke(None, config=...)
    with an updated state from graph.update_state().
    """
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info(
        "graph_hitl_gateway",
        uncertainty_stage=state.get("uncertainty_stage"),
        root_margin=round(state.get("root_margin", 0.0), 3),
        sub_margin=round(state.get("sub_margin", 0.0), 3),
    )

    return {
        "requires_human_review": True,
        "execution_trail": state.get("execution_trail", []) + ["hitl_gateway"],
    }


# ---------------------------------------------------------------------------
# Private helper: Hospital name extraction
# ---------------------------------------------------------------------------


async def _extract_hospital_name(ocr_text: str, config: AppConfig) -> str | None:
    """Extract hospital or clinic name from OCR text.

    Uses a dedicated plain LLM call (no logprobs needed — free text output).
    Extracted separately from classification to keep each node's responsibility
    focused (SRP: classify vs. extract named entities).
    """
    if not ocr_text.strip():
        return None

    llm = _create_llm(config, logprobs=False)
    messages = [
        SystemMessage(content=HOSPITAL_EXTRACTION_SYSTEM),
        HumanMessage(content=HOSPITAL_EXTRACTION_USER.format(ocr_text=ocr_text[:2000])),
    ]

    response = await llm.ainvoke(messages)
    name = response.content.strip()  # type: ignore[union-attr]

    _null_responses = frozenset({"null", "none", "n/a", "unknown", ""})
    if not name or name.lower() in _null_responses:
        return None

    return name
