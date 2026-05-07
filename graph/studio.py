from config.settings import AppConfig
from graph.builder import build_classification_graph, ClassificationDependencies
from ocr.engine import create_di_client
from graph.nodes import get_llm

# Load configuration (pulls from environment or .env file)
config = AppConfig()  # type: ignore[call-arg]

# Setup dependencies for the graph
di_client = create_di_client(config)
def llm_factory(logprobs: bool = False):
    return get_llm(config, logprobs=logprobs)

deps = ClassificationDependencies(
    di_client=di_client,
    ocr_model_id=config.azure_di_model,
    llm_factory=llm_factory,
)

# Compile the graph instance for LangGraph Studio
# LangGraph Studio will automatically wrap this with its own checkpointer
graph = build_classification_graph(deps)
