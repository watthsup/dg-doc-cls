from __future__ import annotations

import structlog
from graph.state import GraphState
from schemas.multi_page import MultiPageResult

log = structlog.get_logger()

def reduce_results_node(state: GraphState) -> GraphState:
    """Aggregate all page results into the final MultiPageResult."""
    results = state.get("page_results", [])
    
    # Sort results by page index to ensure order
    sorted_results = sorted(results, key=lambda x: x.page_index)
    
    # Extract OCR metrics from the graph metrics
    node_metrics = state.get("node_metrics", {})
    ocr_lat = node_metrics.get("ocr_ingestion", {}).get("latency_ms", 0)

    # Create final result
    final_result = MultiPageResult(
        document_id=state.get("document_id", "unknown"),
        file_name=state.get("file_path", "unknown"),
        total_pages=len(sorted_results),
        pages=sorted_results,
        processing_time_ms=0, # Will be set by runner
        pipeline_metrics={"azure_di_ocr_latency_ms": ocr_lat}
    )
    
    return {
        "final_result": final_result,
        "execution_trail": ["reduce_results"]
    }
