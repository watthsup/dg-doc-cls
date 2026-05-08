from __future__ import annotations

import time
from typing import Any, Callable
from src.adapters.orchestration.doc_cls.state import GraphState

NodeFn = Callable[[GraphState], dict[str, Any]]

def make_ocr_node(di_client: Any, model_id: str) -> NodeFn:
    """
    Factory for OCR node.
    Matches the pattern from doc-structure-agent/nodes/ocr.py.
    """
    def ocr_node(state: GraphState) -> dict[str, Any]:
        file_path = state.get("file_path")
        ocr_text = state.get("azure_ocr_text")

        # Skip OCR if text is already provided
        if ocr_text:
            from src.domain.models.document import OCRResult, OCRPageResult
            dummy_page = OCRPageResult(page_index=0, text=ocr_text, mean_confidence=100.0)
            dummy_result = OCRResult(merged_text=ocr_text, pages=[dummy_page], overall_confidence=1.0)
            return {
                "ocr_result": dummy_result,
                "azure_ocr_text": ocr_text,
                "node_metrics": {"ocr_ingestion": {"latency_ms": 0}}
            }

        if not file_path:
            return {"error": "Missing file_path for OCR"}

        start_time = time.perf_counter()
        try:
            from pathlib import Path
            from src.adapters.outbound.ocr.client import analyze_document
            ocr_result = analyze_document(di_client, Path(file_path), model_id)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            return {
                "ocr_result": ocr_result,
                "azure_ocr_text": ocr_result.merged_text,
                "node_metrics": {"ocr_ingestion": {"latency_ms": latency_ms}}
            }
        except Exception as e:
            return {"error": f"OCR failed: {str(e)}"}

    return ocr_node
