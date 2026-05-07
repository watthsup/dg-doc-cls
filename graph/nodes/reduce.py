from __future__ import annotations

import time
from typing import Any
from graph.state import GraphState
from schemas.multi_page import MultiPageResult

def reduce_node(state: GraphState) -> dict[str, Any]:
    """
    Aggregates all page results into the final MultiPageResult.
    Removes global classification logic as requested (focus on per-page).
    """
    results = state.get("page_results", [])
    if not results:
        return {"error": "No page results collected"}

    # Sort by page index to maintain original order
    sorted_results = sorted(results, key=lambda x: x.page_index)
    
    # Calculate total processing time
    start_time = state.get("start_time", time.time())
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Calculate OCR latency from metrics
    node_metrics = state.get("node_metrics", {})
    ocr_lat = node_metrics.get("ocr_ingestion", {}).get("latency_ms", 0)

    # Create final result object
    final_result = MultiPageResult(
        document_id=state.get("document_id", "unknown"),
        file_name=state.get("file_path", "unknown"),
        total_pages=len(sorted_results),
        pages=sorted_results,
        processing_time_ms=elapsed_ms,
        pipeline_metrics={"azure_di_ocr_latency_ms": ocr_lat}
    )
    
    return {
        "final_result": final_result
    }
