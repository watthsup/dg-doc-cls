from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from graph.state import GraphState, create_initial_state
from graph.nodes.ocr import make_ocr_node
from graph.nodes.router import make_root_router_node
from graph.nodes.specialist import make_specialist_node
from graph.nodes.map_dispatch import map_dispatch_node
from graph.nodes.reduce import reduce_results_node


@dataclass(frozen=True, slots=True)
class ClassificationDependencies:
    di_client: Any
    ocr_model_id: str
    llm_factory: Any  # Function taking logprobs: bool


def route_after_root(state: GraphState) -> Literal["med_specialist", "nonmed_specialist"]:
    """Route to appropriate specialist based on root code."""
    root_code = state.get("root_code", "MED")
    if root_code == "MED":
        return "med_specialist"
    return "nonmed_specialist"


def build_classification_graph(
    deps: ClassificationDependencies,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the classification graph using Native Map/Reduce."""
    
    # 1. Main Document Graph
    builder = StateGraph(GraphState)
    
    # Nodes
    builder.add_node("ocr", make_ocr_node(deps.di_client, deps.ocr_model_id))
    builder.add_node("dispatch", map_dispatch_node)
    builder.add_node("reduce", reduce_results_node)
    
    # Main Flow
    builder.add_edge(START, "ocr")
    builder.add_conditional_edges("ocr", map_dispatch_node, ["classify_page"])
    builder.add_edge("classify_page", "reduce")
    builder.add_edge("reduce", END)

    # 2. Page Classification Sub-graph (This is what 'Send' targets)
    # In LangGraph, 'Send' targets a node name. If that node name 
    # refers to another graph or a sequence of nodes, we define it here.
    
    # We define the node 'classify_page' as a mini-flow
    page_builder = StateGraph(GraphState)
    page_builder.add_node("root_router", make_root_router_node(deps.llm_factory))
    page_builder.add_node("med_specialist", make_specialist_node(deps.llm_factory, is_med=True))
    page_builder.add_node("nonmed_specialist", make_specialist_node(deps.llm_factory, is_med=False))
    
    page_builder.add_edge(START, "root_router")
    page_builder.add_conditional_edges(
        "root_router",
        route_after_root,
        {
            "med_specialist": "med_specialist",
            "nonmed_specialist": "nonmed_specialist",
        }
    )
    page_builder.add_edge("med_specialist", END)
    page_builder.add_edge("nonmed_specialist", END)
    
    # Register the page sub-graph as a node in the main graph
    builder.add_node("classify_page", page_builder.compile())

    return builder.compile(checkpointer=checkpointer)


class ClassificationRunner:
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
        file_path: str | None = None,
        ocr_text: str | None = None,
        *,
        document_id: str | None = None,
        thread_id: str | None = None,
    ) -> GraphState:
        import time
        from typing import cast
        
        start_time = time.monotonic()
        
        # Support flexible input (text or file)
        initial_state = create_initial_state(
            document_id=document_id or "unnamed",
            file_path=file_path or "text_input",
            file_type="pdf" if file_path else "text"
        )
        if ocr_text:
            initial_state["azure_ocr_text"] = ocr_text

        config = {"configurable": {"thread_id": thread_id or document_id or "default"}}
        
        result = await self._graph.ainvoke(initial_state, config=config)
        
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        
        # Finalize the result object with total processing time
        if "final_result" in result:
            result["final_result"].processing_time_ms = elapsed_ms
            
        return cast(GraphState, result)
