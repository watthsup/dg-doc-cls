"""LLM adapter — document classifier with structured output.

Supports both Standard OpenAI and Azure OpenAI (Entra ID).
Falls back to Azure if OPENAI_API_KEY is not set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.application.prompts.legacy_prompts import CLASSIFICATION_PROMPT
from src.application.contracts.llm_output import LLMOutput

if TYPE_CHECKING:
    from src.infrastructure.config.settings import AppConfig

logger = structlog.get_logger()


class LLMClassifier:
    """Document classifier using OpenAI with Pydantic structured output.

    Single call returns classification.
    Automatically selects Standard OpenAI or Azure OpenAI based on config.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        if config.openai_api_key:
            logger.info("llm_classifier_using_openai", model=config.openai_model)
            self._llm = ChatOpenAI(
                model=config.openai_model,
                api_key=config.openai_api_key,
                temperature=config.llm_temperature,
                timeout=config.llm_timeout,
            )
        else:
            global _AZURE_TOKEN_PROVIDER
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            logger.info("llm_classifier_using_azure", deployment=config.azure_openai_deployment)
            
            if _AZURE_TOKEN_PROVIDER is None:
                _AZURE_TOKEN_PROVIDER = get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
                )
                
            self._llm = AzureChatOpenAI(
                azure_deployment=config.azure_openai_deployment,
                azure_endpoint=config.azure_openai_endpoint,
                azure_ad_token_provider=_AZURE_TOKEN_PROVIDER,
                api_version=config.azure_openai_api_version,
                temperature=config.llm_temperature,
                timeout=config.llm_timeout,
            )

        # Structured output: LLM returns valid LLMOutput JSON
        self._chain = CLASSIFICATION_PROMPT | self._llm.with_structured_output(
            LLMOutput
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, Exception)),
        reraise=True,
    )
    async def classify(self, ocr_text: str) -> LLMOutput:
        """Classify a document.

        Returns LLMOutput which gets merged with computed signals downstream.
        """
        if not ocr_text.strip():
            return LLMOutput(
                primary_class="non_medical",  # type: ignore[arg-type]
                subcategory="other",  # type: ignore[arg-type]
            )

        result = await self._chain.ainvoke({"ocr_text": ocr_text})

        if not isinstance(result, LLMOutput):
            raise TypeError(
                f"Expected LLMOutput, got {type(result).__name__}: {result}"
            )

        return result

_AZURE_TOKEN_PROVIDER = None

def get_llm(config: AppConfig, logprobs: bool = False):
    """Factory for standard/Azure LLMs without structured output.
    
    Caches the Azure token provider to avoid expensive DefaultAzureCredential initialization.
    """
    global _AZURE_TOKEN_PROVIDER
    
    kwargs = {}
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = config.logprobs_top_n

    if config.openai_api_key:
        return ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            temperature=config.llm_temperature,
            timeout=config.llm_timeout,
            **kwargs
        )
    else:
        if _AZURE_TOKEN_PROVIDER is None:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            _AZURE_TOKEN_PROVIDER = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            
        return AzureChatOpenAI(
            azure_deployment=config.azure_openai_deployment,
            azure_endpoint=config.azure_openai_endpoint,
            azure_ad_token_provider=_AZURE_TOKEN_PROVIDER,
            api_version=config.azure_openai_api_version,
            temperature=config.llm_temperature,
            timeout=config.llm_timeout,
            **kwargs
        )
