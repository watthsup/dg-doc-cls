"""Batch processor — concurrent document processing with error isolation."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from classifier.llm import LLMClassifier
from ocr.engine import create_di_client
from pipeline.document import process_document
from schemas.models import BatchResult, DocumentError, DocumentResult

if TYPE_CHECKING:
    from config.settings import AppConfig
    from schemas.models import DocumentInput

logger = structlog.get_logger()


async def process_batch(
    documents: list[DocumentInput],
    config: AppConfig,
) -> BatchResult:
    """Process a batch of documents with controlled concurrency.

    Shares a single Azure DI client and LLM classifier across all
    documents for connection reuse. Failures are captured per-document.
    """
    log = logger.bind(batch_size=len(documents))
    log.info("batch_started")
    start_time = time.monotonic()

    # Shared across the batch (stateless — safe for concurrent use)
    classifier = LLMClassifier(config)
    di_client = create_di_client(config)

    semaphore = asyncio.Semaphore(config.max_concurrency)
    results: list[DocumentResult] = []
    errors: list[DocumentError] = []

    async def _process_one(doc: DocumentInput) -> DocumentResult | DocumentError:
        async with semaphore:
            try:
                return await process_document(
                    doc, classifier, config, di_client=di_client
                )
            except Exception as e:
                error_type = type(e).__name__
                log.error(
                    "document_failed",
                    document_id=doc.document_id,
                    error_type=error_type,
                    error=str(e),
                )
                return DocumentError(
                    document_id=doc.document_id,
                    error_type=error_type,
                    error_message=str(e),
                    stage="pipeline",
                )

    tasks = [_process_one(doc) for doc in documents]
    outcomes = await asyncio.gather(*tasks)

    for outcome in outcomes:
        if isinstance(outcome, DocumentResult):
            results.append(outcome)
        else:
            errors.append(outcome)

    elapsed_s = time.monotonic() - start_time
    log.info(
        "batch_complete",
        successful=len(results),
        failed=len(errors),
        elapsed_seconds=round(elapsed_s, 1),
    )

    return BatchResult(
        total_documents=len(documents),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )
