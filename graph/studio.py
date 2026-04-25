from config.settings import AppConfig
from graph.builder import build_classification_graph

# Load configuration (pulls from environment or .env file)
config = AppConfig()

# Compile the graph instance for LangGraph Studio to load
graph = build_classification_graph(config)
