"""Application configuration via Pydantic Settings.

All Azure-specific settings (Document Intelligence, OpenAI Foundry)
are loaded from environment variables or .env file.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(StrEnum):
    """Log output format — console for development, json for production."""

    CONSOLE = "console"
    JSON = "json"


class AppConfig(BaseSettings):
    """Central configuration for the document classification system.

    All settings can be overridden via environment variables.
    SecretStr is used for API keys to prevent accidental logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Azure Document Intelligence ---
    azure_di_endpoint: str = Field(
        description="Azure Document Intelligence endpoint URL",
    )
    azure_di_key: SecretStr = Field(
        description="Azure Document Intelligence API key",
    )
    azure_di_model: str = Field(
        default="prebuilt-layout",
        description="Azure DI model ID (prebuilt-layout or prebuilt-read)",
    )

    # --- Azure OpenAI (Foundry) ---
    azure_openai_endpoint: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL (e.g. https://xxx.openai.azure.com/)",
    )
    azure_openai_deployment: str | None = Field(
        default=None,
        description="Azure OpenAI deployment name",
    )
    azure_openai_api_version: str = Field(
        default="2024-05-01-preview",
        description="Azure OpenAI API version",
    )

    # --- Standard OpenAI ---
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="Standard OpenAI API key (if provided, overrides Azure)",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name to use if openai_api_key is provided",
    )

    # --- LLM Settings ---
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Zero temperature for deterministic classification",
    )
    llm_timeout: int = Field(default=30, ge=5, le=120)
    llm_max_retries: int = Field(default=3, ge=1, le=10)

    # --- Page Selection ---
    max_pages: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum pages to OCR per document",
    )

    # --- Image Quality Thresholds ---
    blur_threshold: float = Field(
        default=100.0,
        ge=0.0,
        description="Laplacian variance below this = blurry (tune empirically)",
    )
    contrast_min: float = Field(
        default=30.0,
        ge=0.0,
        description="Grayscale std dev below this = low contrast",
    )

    # --- Batch Processing ---
    max_concurrency: int = Field(default=5, ge=1, le=50)

    # --- Confidence Weights ---
    ocr_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    quality_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    # --- Confidence Thresholds ---
    low_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Documents below this are flagged for review",
    )

    # --- LangGraph HITL ---
    margin_threshold: float = Field(
        default=1.5,
        ge=0.0,
        description="Logprob margin below this triggers HITL interrupt",
    )
    checkpoint_db_path: str = Field(
        default="./checkpoints/doc_cls.db",
        description="SQLite checkpoint database path for LangGraph state persistence",
    )
    logprobs_top_n: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of top logprobs to capture from LLM",
    )

    # --- LangSmith Tracing ---
    langsmith_api_key: SecretStr | None = Field(default=None)
    langsmith_tracing: bool = Field(default=False)
    langsmith_project: str = Field(default="docguru-cls")

    # --- Logging ---
    log_format: LogFormat = Field(default=LogFormat.CONSOLE)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # --- Output ---
    output_format: Literal["jsonl", "csv", "both"] = Field(default="jsonl")
