from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.state import GraphState, create_initial_state
from graph.nodes.ocr import make_ocr_node
from graph.nodes.map_dispatch import dispatch_pages
from graph.nodes.classify_page import make_classify_page_node
from graph.nodes.reduce import reduce_node
from schemas.multi_page import MultiPageResult


@dataclass(frozen=True, slots=True)
class ClassificationDependencies:
    di_client: Any
    ocr_model_id: str
    llm_factory: Any  # Function taking logprobs: bool


def build_classification_graph(
    deps: ClassificationDependencies,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """
    Build the classification graph.
    Matches the pattern from doc-structure-agent/builder.py.
    """
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("ocr", make_ocr_node(deps.di_client, deps.ocr_model_id))
    graph.add_node("classify_page", make_classify_page_node(deps.llm_factory))
    graph.add_node("reduce", reduce_node)

    # Main Flow
    graph.add_edge(START, "ocr")
    graph.add_conditional_edges("ocr", dispatch_pages, ["classify_page"])
    graph.add_edge("classify_page", "reduce")
    graph.add_edge("reduce", END)

    return graph.compile(checkpointer=checkpointer)


class ClassificationRunner:
    """
    Runner for the classification pipeline.
    Matches the pattern from doc-structure-agent ParallelExtractRunner.
    """
    def __init__(
        self,
        deps: ClassificationDependencies,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._graph = build_classification_graph(deps, checkpointer=checkpointer)
        self._has_checkpointer = checkpointer is not None

    @classmethod
    def from_env(
        cls, *, checkpointer: BaseCheckpointSaver | None = None
    ) -> ClassificationRunner:
        from config.settings import AppConfig
        from ocr.engine import create_di_client
        from graph.nodes import get_llm
        
        config = AppConfig()  # type: ignore[call-arg]
        di_client = create_di_client(config)
        
        def llm_factory(logprobs: bool = False):
            return get_llm(config, logprobs=logprobs)

        deps = ClassificationDependencies(
            di_client=di_client,
            ocr_model_id=config.azure_di_model,
            llm_factory=llm_factory,
        )
        return cls(deps, checkpointer=checkpointer)

    async def run(
        self,
        file_path: str | Path | None = None,
        ocr_text: str | None = None,
        document_id: str | None = None,
        config: dict | None = None,
    ) -> MultiPageResult:
        """
        Run the classification graph and return the aggregated result.
        """
        # Ensure we have a valid file_path string for create_initial_state
        str_path = str(file_path) if file_path else "raw_text"
        
        initial_state = create_initial_state(str_path, document_id=document_id)
        if ocr_text:
            initial_state["azure_ocr_text"] = ocr_text
        
        # Build graph config
        if self._has_checkpointer and (not config or "configurable" not in config):
            config = {"configurable": {"thread_id": initial_state["document_id"]}}
            
        result = await self._graph.ainvoke(initial_state, config=config)
        
        if not isinstance(result, dict) or "final_result" not in result:
            error_msg = result.get("error") if isinstance(result, dict) else "Unknown error"
            raise RuntimeError(f"Pipeline failed: {error_msg}")
            
        return result["final_result"]
