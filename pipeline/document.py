"""Per-document classification pipeline.

Flow:
  1. Load document → PIL Images (local, for OpenCV quality assessment)
  2. Sample pages
  3. OpenCV quality assessment (blur, skew, contrast) → image_quality
  4. Send file to Azure DI → OCR text + word confidence
  5. merge_quality(image_quality, ocr_result) → final QualityAssessment
  6. LLM classification (structured output → LLMOutput)
  7. Keyword score for classified category
  8. Confidence calculation from all signals
  9. Merge everything → DocumentResult
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from classifier.keywords import get_keyword_score_for_class
from confidence.calculator import ConfidenceWeights, calculate_confidence
from ocr.engine import analyze_document, create_di_client, images_to_numpy, load_document_images
from ocr.page_sampler import sample_pages
from ocr.quality import assess_multi_page_quality, merge_quality
from schemas.models import (
    ClassificationResult,
    DocumentInput,
    DocumentResult,
    ProcessingMetadata,
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
    """Process a single document through the full pipeline.

    Quality assessment runs locally on rendered images.
    Azure DI receives raw file bytes for OCR.
    Both are independent — could be parallelized if needed.
    """
    start_time = time.monotonic()
    log = logger.bind(document_id=document.document_id)

    if di_client is None:
        di_client = create_di_client(config)

    # --- Load pages locally (PIL Images for quality assessment) ---
    log.info("loading_document", file_type=document.file_type)
    all_images = load_document_images(
        file_path=document.file_path,
        file_type=document.file_type,
    )
    total_pages = len(all_images)

    # --- Extract doc_type from filename ---
    # Pattern: T232064267_263302157_U62400.tiff -> U62400 is the 3rd part
    filename_doc_type = None
    parts = document.file_path.stem.split("_")
    if len(parts) >= 3:
        filename_doc_type = parts[2]

    page_indices = sample_pages(
        total_pages=total_pages, max_pages=config.max_pages
    )
    selected_images = [all_images[i] for i in page_indices]
    log.info("pages_selected", total=total_pages, selected=page_indices)

    # --- OpenCV quality assessment (local, before Azure DI) ---
    numpy_images = images_to_numpy(selected_images)
    image_quality = assess_multi_page_quality(
        images=numpy_images,
        blur_threshold=config.blur_threshold,
        contrast_min=config.contrast_min,
    )
    log.info(
        "opencv_quality",
        image_quality=round(image_quality.quality_score, 3),
        skew_angle=round(image_quality.skew_angle, 1),
        issues=len(image_quality.issues),
    )

    # --- Azure DI OCR (raw file bytes → text + word confidence) ---
    log.info("starting_ocr")
    ocr_result = analyze_document(
        client=di_client,
        file_path=document.file_path,
        model_id=config.azure_di_model,
        page_indices=page_indices,
    )
    log.info(
        "ocr_complete",
        text_length=len(ocr_result.merged_text),
        ocr_confidence=round(ocr_result.overall_confidence, 3),
        pages_returned=len(ocr_result.pages),
    )

    # --- Merge OpenCV + OCR confidence into final quality assessment ---
    quality = merge_quality(image_quality, ocr_result)
    log.info(
        "quality_assessment_complete",
        image_quality=round(quality.quality_score, 3),
        ocr_confidence=round(ocr_result.overall_confidence, 3),
        total_issues=len(quality.issues),
    )

    # --- LLM: single call → classification + hospital_name ---
    log.info("classifying")
    llm_output = await classifier.classify(ocr_result.merged_text)
    log.info(
        "classified",
        primary=llm_output.primary_class.value,
        sub=llm_output.subcategory.value,
        hospital=llm_output.hospital_name,
    )

    # --- Keyword score for the CLASSIFIED category ---
    keyword_score = get_keyword_score_for_class(
        ocr_result.merged_text, llm_output.subcategory
    )

    # --- Confidence: weighted composite of all 3 signals ---
    signals = SignalScores(
        ocr_confidence=ocr_result.overall_confidence,
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

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    actual_pages = [p.page_index for p in ocr_result.pages]

    result = DocumentResult(
        document_id=document.document_id,
        filename_doc_type=filename_doc_type,
        classification=ClassificationResult(
            primary_class=llm_output.primary_class,
            subcategory=llm_output.subcategory,
            hospital_name=llm_output.hospital_name,
        ),
        confidence=confidence,
        signals=signals,
        quality_assessment=quality,
        ocr_text=ocr_result.merged_text,
        processing_metadata=ProcessingMetadata(
            pages_used=actual_pages,
            total_pages=total_pages,
            processing_time_ms=elapsed_ms,
        ),
    )

    if confidence < config.low_confidence_threshold:
        log.warning(
            "low_confidence",
            confidence=round(confidence, 3),
            threshold=config.low_confidence_threshold,
        )

    log.info("document_complete", confidence=round(confidence, 3), elapsed_ms=elapsed_ms)
    return result
