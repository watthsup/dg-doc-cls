"""LangGraph node functions for the hierarchical classification pipeline.

Each node receives and returns a GraphState dict.
Nodes are pure functions — all side-effects (LLM calls, OCR) are explicit.
"""

from __future__ import annotations

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


def _get_config() -> AppConfig:
    """Lazily load configuration."""
    return AppConfig()  # type: ignore[call-arg]


def _create_llm_with_logprobs(config: AppConfig) -> AzureChatOpenAI:
    """Create an AzureChatOpenAI instance with logprobs enabled.

    Uses raw LLM calls (NOT with_structured_output) to preserve
    response_metadata containing logprob distributions.
    """
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureChatOpenAI(
        azure_deployment=config.azure_openai_deployment,
        azure_endpoint=config.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=config.azure_openai_api_version,
        temperature=0.0,
        timeout=config.llm_timeout,
        model_kwargs={
            "logprobs": True,
            "top_logprobs": config.logprobs_top_n,
        },
    )


# ---------------------------------------------------------------------------
# Node 0: Azure OCR Ingestion
# ---------------------------------------------------------------------------


def ocr_ingestion_node(state: GraphState) -> dict[str, Any]:
    """Call Azure Document Intelligence and store results in state.

    This node runs the same OCR engine used by the existing pipeline
    but stores results in the GraphState format.
    """
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_ocr_ingestion_start")

    config = _get_config()
    di_client = create_di_client(config)

    file_path = Path(state["file_path"])

    ocr_result = analyze_document(
        client=di_client,
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
    """Classify document into MED or NON using single-token logprob analysis."""
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_root_router_start")

    config = _get_config()
    llm = _create_llm_with_logprobs(config)

    ocr_text = state.get("azure_ocr_text", "")

    messages = [
        SystemMessage(content=ROOT_ROUTER_SYSTEM),
        HumanMessage(content=ROOT_ROUTER_USER.format(ocr_text=ocr_text[:4000])),
    ]

    response = await llm.ainvoke(messages)

    # Extract the generated token (should be "MED" or "NON")
    raw_code = response.content.strip().upper()  # type: ignore[union-attr]
    valid_tokens = [c.value for c in VALID_ROOT_CODES]

    # Analyze logprobs
    analysis = analyze_logprobs(response.response_metadata, valid_tokens)

    # Validate code
    root_code = raw_code if raw_code in valid_tokens else analysis.top1_token
    if root_code not in valid_tokens:
        root_code = "MED"  # Safe default

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
    """Sub-classify medical document into specific type."""
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_med_specialist_start")

    config = _get_config()
    llm = _create_llm_with_logprobs(config)

    ocr_text = state.get("azure_ocr_text", "")

    messages = [
        SystemMessage(content=MED_SPECIALIST_SYSTEM),
        HumanMessage(content=MED_SPECIALIST_USER.format(ocr_text=ocr_text[:4000])),
    ]

    response = await llm.ainvoke(messages)

    raw_code = response.content.strip().upper()  # type: ignore[union-attr]
    valid_tokens = [c.value for c in VALID_MED_SUB_CODES]

    analysis = analyze_logprobs(response.response_metadata, valid_tokens)

    sub_code = raw_code if raw_code in valid_tokens else analysis.top1_token
    if sub_code not in valid_tokens:
        sub_code = "MOT"  # Medical other fallback

    is_uncertain = analysis.margin_score < config.margin_threshold

    # Extract hospital name for medical documents
    hospital_name = await _extract_hospital_name(ocr_text, config)

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
        "is_uncertain": is_uncertain or state.get("is_uncertain", False),
        "uncertainty_stage": "sub" if is_uncertain else state.get("uncertainty_stage"),
        "hospital_name": hospital_name,
        "execution_trail": state.get("execution_trail", []) + ["med_specialist"],
    }


# ---------------------------------------------------------------------------
# Node 2b: Non-Medical Specialist
# ---------------------------------------------------------------------------


async def nonmed_specialist_node(state: GraphState) -> dict[str, Any]:
    """Sub-classify non-medical document into specific type."""
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info("graph_nonmed_specialist_start")

    config = _get_config()
    llm = _create_llm_with_logprobs(config)

    ocr_text = state.get("azure_ocr_text", "")

    messages = [
        SystemMessage(content=NONMED_SPECIALIST_SYSTEM),
        HumanMessage(content=NONMED_SPECIALIST_USER.format(ocr_text=ocr_text[:4000])),
    ]

    response = await llm.ainvoke(messages)

    raw_code = response.content.strip().upper()  # type: ignore[union-attr]
    valid_tokens = [c.value for c in VALID_NONMED_SUB_CODES]

    analysis = analyze_logprobs(response.response_metadata, valid_tokens)

    sub_code = raw_code if raw_code in valid_tokens else analysis.top1_token
    if sub_code not in valid_tokens:
        sub_code = "OTH"  # Non-medical other fallback

    is_uncertain = analysis.margin_score < config.margin_threshold

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
        "hospital_name": None,
        "execution_trail": state.get("execution_trail", []) + ["nonmed_specialist"],
    }


# ---------------------------------------------------------------------------
# Node 3: HITL Gateway
# ---------------------------------------------------------------------------


def hitl_gateway_node(state: GraphState) -> dict[str, Any]:
    """Mark state for human review and persist via checkpointer.

    This node only flags the state — actual interruption is handled
    by the LangGraph `interrupt_before` mechanism at compile time.
    """
    log = logger.bind(document_id=state.get("document_id", "unknown"))
    log.info(
        "graph_hitl_gateway",
        uncertainty_stage=state.get("uncertainty_stage"),
        root_margin=state.get("root_margin", 0.0),
        sub_margin=state.get("sub_margin", 0.0),
    )

    return {
        "requires_human_review": True,
        "execution_trail": state.get("execution_trail", []) + ["hitl_gateway"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _extract_hospital_name(ocr_text: str, config: AppConfig) -> str | None:
    """Extract hospital name using a separate LLM call."""
    if not ocr_text.strip():
        return None

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    llm = AzureChatOpenAI(
        azure_deployment=config.azure_openai_deployment,
        azure_endpoint=config.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=config.azure_openai_api_version,
        temperature=0.0,
        timeout=config.llm_timeout,
    )

    messages = [
        SystemMessage(content=HOSPITAL_EXTRACTION_SYSTEM),
        HumanMessage(content=HOSPITAL_EXTRACTION_USER.format(ocr_text=ocr_text[:2000])),
    ]

    response = await llm.ainvoke(messages)
    name = response.content.strip()  # type: ignore[union-attr]

    if not name or name.lower() in ("null", "none", "n/a", "unknown", ""):
        return None

    return name
