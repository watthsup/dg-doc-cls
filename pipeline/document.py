"""Per-document classification pipeline.

Flow:
  1. Load document → PIL Images (all pages)
  2. OpenCV quality assessment per page → list of image_quality
  3. Send file to Azure DI → OCR text + word confidence for all pages
  4. Concurrently process each page:
     a. merge_quality(image_quality, ocr_page)
     b. LLM classification
     c. Keyword score
     d. Calculate page confidence
     e. Return PageResult
  5. Aggregate PageResults to determine document-level highest confidence result
  6. Return DocumentResult
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from classifier.keywords import get_keyword_score_for_class
from confidence.calculator import ConfidenceWeights, calculate_confidence
from ocr.engine import analyze_document, create_di_client, images_to_numpy, load_document_images
from ocr.quality import assess_multi_page_quality, merge_quality
from schemas.models import (
    ClassificationResult,
    DocumentInput,
    DocumentResult,
    PageResult,
    ProcessingMetadata,
    QualityAssessment,
    SignalScores,
)

if TYPE_CHECKING:
    from azure.ai.documentintelligence import DocumentIntelligenceClient

    from classifier.llm import LLMClassifier
    from config.settings import AppConfig

logger = structlog.get_logger()


async def process_document(
    document: DocumentInput,
    classifier: LLMClassifier,
    config: AppConfig,
    di_client: DocumentIntelligenceClient | None = None,
) -> DocumentResult:
    """Process all pages of a document through the classification pipeline."""
    start_time = time.monotonic()
    log = logger.bind(document_id=document.document_id)

    if di_client is None:
        di_client = create_di_client(config)

    # --- 1. Load pages locally (PIL Images for quality assessment) ---
    log.info("loading_document", file_type=document.file_type)
    all_images = load_document_images(
        file_path=document.file_path,
        file_type=document.file_type,
    )
    total_pages = len(all_images)

    # --- 2. Extract doc_type from filename ---
    filename_doc_type = None
    parts = document.file_path.stem.split("_")
    if len(parts) >= 3:
        filename_doc_type = parts[2]

    # --- 3. OpenCV quality assessment (per page, local) ---
    numpy_images = images_to_numpy(all_images)
    image_qualities = assess_multi_page_quality(
        images=numpy_images,
        blur_threshold=config.blur_threshold,
        contrast_min=config.contrast_min,
    )

    # --- 4. Azure DI OCR (all pages) ---
    log.info("starting_ocr", total_pages=total_pages)
    ocr_result = analyze_document(
        client=di_client,
        file_path=document.file_path,
        model_id=config.azure_di_model,
        page_indices=None,  # Process all pages
    )
    log.info(
        "ocr_complete",
        text_length=len(ocr_result.merged_text),
        ocr_confidence=round(ocr_result.overall_confidence, 3),
        pages_returned=len(ocr_result.pages),
    )

    ocr_pages_map = {p.page_index: p for p in ocr_result.pages}

    # --- 5. Concurrent per-page processing ---
    async def process_single_page(page_idx: int) -> PageResult | None:
        if page_idx not in ocr_pages_map:
            return None

        ocr_page = ocr_pages_map[page_idx]
        image_quality = (
            image_qualities[page_idx]
            if page_idx < len(image_qualities)
            else QualityAssessment()
        )

        # Merge quality
        quality = merge_quality(image_quality, ocr_page)

        # LLM Classification
        llm_output = await classifier.classify(ocr_page.text)

        # Keyword score
        keyword_score = get_keyword_score_for_class(
            ocr_page.text, llm_output.subcategory
        )

        # Confidence
        signals = SignalScores(
            ocr_confidence=ocr_page.mean_confidence / 100.0,
            keyword_score=keyword_score,
            quality_score=quality.quality_score,
        )
        confidence = calculate_confidence(
            signals,
            weights=ConfidenceWeights(
                keyword=config.keyword_weight,
                ocr=config.ocr_weight,
                quality=config.quality_weight,
            ),
        )

        return PageResult(
            page_index=page_idx,
            classification=ClassificationResult(
                primary_class=llm_output.primary_class,
                subcategory=llm_output.subcategory,
                hospital_name=llm_output.hospital_name,
            ),
            confidence=confidence,
            signals=signals,
            quality_assessment=quality,
            ocr_text=ocr_page.text,
        )

    log.info("classifying_pages_concurrently")
    page_tasks = [process_single_page(i) for i in range(total_pages)]
    page_results_raw = await asyncio.gather(*page_tasks)

    # Filter out missing pages
    page_results = [r for r in page_results_raw if r is not None]

    if not page_results:
        raise ValueError(f"No pages could be processed for document {document.document_id}")

    # --- 6. Aggregate to DocumentResult ---
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    result = DocumentResult(
        document_id=document.document_id,
        filename_doc_type=filename_doc_type,
        pages=page_results,
        processing_metadata=ProcessingMetadata(
            pages_used=[p.page_index for p in page_results],
            total_pages=total_pages,
            processing_time_ms=elapsed_ms,
        ),
    )

    log.info(
        "document_complete",
        total_pages=total_pages,
        processed_pages=len(page_results),
        elapsed_ms=elapsed_ms,
    )
    return result
