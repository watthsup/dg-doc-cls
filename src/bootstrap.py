from __future__ import annotations

import structlog
from typing import Any

from src.infrastructure.config.settings import AppConfig
from src.adapters.orchestration.doc_cls.builder import ClassificationRunner, ClassificationDependencies
from src.adapters.orchestration.vanilla.builder import build_vanilla_runner, VanillaRunner
from src.application.use_cases.classify_page_service import ClassifyPageService
from src.adapters.outbound.llm.client import get_llm
from src.adapters.outbound.ocr.client import create_di_client

log = structlog.get_logger()

def build_classification_runner(
    config: AppConfig | None = None, 
    graph_type: str = "doc_cls"
) -> Any:
    """Composition root: builds the classification graph runner with all dependencies."""
    if config is None:
        config = AppConfig()  # type: ignore[call-arg]
    
    if graph_type == "vanilla":
        return build_vanilla_runner(config)
    
    if graph_type != "doc_cls":
        log.warning("unsupported_graph_type", graph_type=graph_type)
        
    di_client = create_di_client(config)
    
    def llm_factory(logprobs: bool = False):
        return get_llm(config, logprobs=logprobs)
        
    classifier = ClassifyPageService(llm_factory=llm_factory)
    
    deps = ClassificationDependencies(
        di_client=di_client,
        ocr_model_id=config.azure_di_model,
        classifier=classifier,
    )
    
    return ClassificationRunner(deps)
