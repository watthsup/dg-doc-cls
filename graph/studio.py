from config.settings import AppConfig
from graph.builder import build_classification_graph

# Load configuration (pulls from environment or .env file)
config = AppConfig()

# Compile the graph instance without a custom checkpointer,
# because LangGraph Studio injects its own Postgres checkpointer automatically.
graph = build_classification_graph(config, use_checkpointer=False)
