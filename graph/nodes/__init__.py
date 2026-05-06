from __future__ import annotations

import structlog
from typing import Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from config.settings import AppConfig

log = structlog.get_logger()

def get_llm(config: AppConfig, logprobs: bool = False) -> Any:
    """Factory to create the LLM instance based on environment config."""
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "timeout": config.llm_timeout,
        "max_retries": config.llm_max_retries,
    }
    
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = config.logprobs_top_n

    if config.openai_api_key:
        return ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            **kwargs
        )
    
    return AzureChatOpenAI(
        azure_deployment=config.azure_openai_deployment,
        api_version=config.azure_openai_api_version,
        azure_endpoint=config.azure_openai_endpoint,
        **kwargs
    )
