"""OCR engine — Azure Document Intelligence adapter + local image loading.

Two responsibilities:
1. load_document_images() — renders PDF/image locally into PIL Images
   (used by quality.py for OpenCV analysis BEFORE Azure DI)
2. analyze_document() — sends file bytes to Azure DI for OCR text + confidence

Keeping image loading here isolates all document I/O in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from pdf2image import convert_from_path
from PIL import Image, ImageSequence

from schemas.models import OCRPageResult, OCRResult, OCRWordResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from config.settings import AppConfig


def load_document_images(
    file_path: Path,
    file_type: str,
    dpi: int = 150,
) -> list[Image.Image]:
    """Load document as PIL Images for local quality assessment.

    Called BEFORE Azure DI — these images are only used for OpenCV quality
    checks (blur, skew, contrast). Azure DI receives raw file bytes separately.

    Supports PDF and multi-page TIFFs natively.
    DPI 150 is sufficient for quality metrics (lower than OCR-grade 300 DPI)
    and keeps memory usage reasonable for batch processing.

    Args:
        file_path: Absolute path to the document file.
        file_type: "pdf" or "image".
        dpi: Render DPI (quality assessment only, not OCR).

    Raises:
        FileNotFoundError: If file_path does not exist.
        RuntimeError: If PDF rendering fails (poppler not installed).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if file_type == "pdf":
        try:
            return convert_from_path(str(file_path), dpi=dpi)
        except Exception as e:
            raise RuntimeError(
                f"PDF rendering failed: {file_path}. "
                f"Ensure poppler is installed (brew install poppler). Error: {e}"
            ) from e
    else:
        # ImageSequence handles both single-page (jpg/png) and multi-page (tiff)
        with Image.open(file_path) as img:
            return [frame.copy() for frame in ImageSequence.Iterator(img)]


def images_to_numpy(images: list[Image.Image]) -> list[NDArray[np.uint8]]:
    """Convert PIL Images to numpy arrays for OpenCV processing.

    Ensures images are converted to RGB first to avoid boolean arrays
    from 1-bit (black & white) source images which OpenCV doesn't support.
    """
    return [np.array(img.convert("RGB")) for img in images]


def create_di_client(config: AppConfig) -> DocumentIntelligenceClient:
    """Create an Azure Document Intelligence client from config."""
    return DocumentIntelligenceClient(
        endpoint=config.azure_di_endpoint,
        credential=AzureKeyCredential(config.azure_di_key.get_secret_value()),
    )


def _pages_to_azure_string(page_indices: list[int]) -> str:
    """Convert 0-indexed page indices to Azure DI page string (1-indexed).

    Examples:
        [0, 1] → "1-2"
        [0] → "1"
        [0, 2, 3] → "1,3-4"
    """
    if not page_indices:
        return "1"

    # Convert to 1-indexed
    pages_1indexed = sorted(p + 1 for p in page_indices)

    # Group consecutive numbers into ranges
    ranges: list[str] = []
    start = pages_1indexed[0]
    end = start

    for page in pages_1indexed[1:]:
        if page == end + 1:
            end = page
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = page
            end = page

    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def analyze_document(
    client: DocumentIntelligenceClient,
    file_path: Path,
    model_id: str = "prebuilt-layout",
    page_indices: list[int] | None = None,
) -> OCRResult:
    """Analyze a document using Azure Document Intelligence.

    Sends raw file bytes — Azure DI handles PDF/image natively.
    No local rendering, no language config (auto-detected).

    Args:
        client: Azure DI client instance.
        file_path: Path to the document file.
        model_id: Azure DI model (prebuilt-layout or prebuilt-read).
        page_indices: 0-indexed page indices to analyze (None = all pages).

    Returns:
        Structured OCR result with text and word confidences.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Build analysis parameters
    kwargs: dict[str, object] = {}
    if page_indices is not None:
        kwargs["pages"] = _pages_to_azure_string(page_indices)

    poller = client.begin_analyze_document(
        model_id=model_id,
        body=file_bytes,
        content_type="application/octet-stream",
        **kwargs,
    )
    result = poller.result()

    return _parse_analyze_result(result)


def _parse_analyze_result(result: object) -> OCRResult:
    """Parse Azure DI AnalyzeResult into our OCR models.

    Azure DI returns word-level confidence as 0–1 float (already normalized).
    """
    pages: list[OCRPageResult] = []

    if not hasattr(result, "pages") or not result.pages:  # type: ignore[union-attr]
        return OCRResult(pages=[], merged_text="", overall_confidence=0.0)

    for page in result.pages:  # type: ignore[union-attr]
        page_index = page.page_number - 1  # Azure DI is 1-indexed

        words: list[OCRWordResult] = []
        text_parts: list[str] = []

        if page.words:
            for word in page.words:
                content = word.content.strip()
                if not content:
                    continue
                # Azure DI confidence is already 0–1
                conf = float(word.confidence) if word.confidence is not None else 0.0
                words.append(OCRWordResult(text=content, confidence=conf * 100))
                text_parts.append(content)

        full_text = " ".join(text_parts)

        # Mean confidence on 0–100 scale (matching OCRPageResult contract)
        if words:
            mean_conf = sum(w.confidence for w in words) / len(words)
        else:
            mean_conf = 0.0

        pages.append(
            OCRPageResult(
                page_index=page_index,
                text=full_text,
                words=words,
                mean_confidence=mean_conf,
            )
        )

    return merge_ocr_results(pages)


def merge_ocr_results(pages: list[OCRPageResult]) -> OCRResult:
    """Merge multiple page OCR results into a single document result.

    Overall confidence is normalized from 0–100 to 0–1.
    """
    if not pages:
        return OCRResult(pages=[], merged_text="", overall_confidence=0.0)

    merged_text = "\n\n".join(p.text for p in pages)
    mean_conf_100 = sum(p.mean_confidence for p in pages) / len(pages)
    overall_confidence = mean_conf_100 / 100.0

    return OCRResult(
        pages=pages,
        merged_text=merged_text,
        overall_confidence=min(1.0, max(0.0, overall_confidence)),
    )
