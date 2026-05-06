from __future__ import annotations

import time
from typing import Any
import structlog

from graph.state import GraphState
from ocr.engine import analyze_document

log = structlog.get_logger()

def make_ocr_node(di_client: Any, model_id: str):
    """Factory for OCR node."""
    def ocr_node(state: GraphState) -> GraphState:
        # Check if text was already provided (flexible input)
        if state.get("azure_ocr_text"):
            log.info("graph_ocr_skipped", reason="text_provided")
            return {
                "execution_trail": ["ocr_ingestion (skipped)"],
                "node_metrics": {"ocr_ingestion": {"latency_ms": 0, "skipped": True}}
            }

        from pathlib import Path
        file_path = state.get("file_path")
        if not file_path:
            return {"error": "Missing file_path for OCR"}

        start_time = time.perf_counter()
        try:
            ocr_result = analyze_document(
                client=di_client,
                file_path=Path(file_path),
                model_id=model_id
            )
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            return {
                "ocr_result": ocr_result,
                "azure_ocr_text": ocr_result.merged_text,
                "execution_trail": ["ocr_ingestion"],
                "node_metrics": {"ocr_ingestion": {"latency_ms": latency_ms}}
            }
        except Exception as e:
            log.error("ocr_failed", error=str(e))
            return {"error": f"OCR failed: {str(e)}"}

    return ocr_node
