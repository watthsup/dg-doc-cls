from __future__ import annotations

from langgraph.types import Send
from graph.state import GraphState

def map_dispatch_node(state: GraphState) -> list[Send]:
    """Split the document into pages and dispatch each for classification."""
    # Note: In our current setup, ocr_result is passed as text segments or ocr_result object
    # If text was provided directly, we wrap it as a single-page task if it's not already split
    
    ocr_result = state.get("ocr_result")
    
    # Handle the case where we already have the text (either from DocumentProcessor or direct input)
    # For this refactor, we assume ocr_result contains the pages if it's a multi-page doc.
    # If ocr_result is missing but azure_ocr_text is present, it's a single-task input.
    
    if not ocr_result:
        # Single page / direct text fallback
        return [Send("classify_page", {
            "azure_ocr_text": state.get("azure_ocr_text", ""),
            "document_id": state.get("document_id"),
            "file_path": state.get("file_path"),
            "file_type": state.get("file_type"),
        })]

    sends = []
    # If ocr_result is from Azure DI, it has a .pages attribute
    for i, page in enumerate(ocr_result.pages):
        sends.append(Send("classify_page", {
            "azure_ocr_text": page.text,
            "document_id": f"{state['document_id']}_p{i}",
            "file_path": state["file_path"],
            "file_type": state["file_type"],
            # Metadata to help reduction later
            "page_index": i,
        }))
        
    return sends
