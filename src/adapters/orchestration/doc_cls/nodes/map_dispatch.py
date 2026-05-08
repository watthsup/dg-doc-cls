from __future__ import annotations

from langgraph.types import Send
from src.adapters.orchestration.doc_cls.state import GraphState, PageState

def dispatch_pages(state: GraphState) -> list[Send]:
    """
    Fan-out to parallel classification branches.
    Matches the pattern from doc-structure-agent/nodes/map_dispatch.py.
    """
    if state.get("error"):
        return []

    ocr_result = state.get("ocr_result")
    if not ocr_result:
        return []

    sends = []
    # Loop through pages and prepare Send objects
    for i, page in enumerate(ocr_result.pages):
        if page.text.strip():
            page_state = PageState(
                page_index=i,
                page_text=page.text,
                document_id=state.get("document_id", "unknown")
            )
            # 'classify_page' is the name of the node in the parent graph
            sends.append(Send("classify_page", page_state))

    return sends
