"""CLI entry point for single document classification via LangGraph pipeline.

Usage:
    python scripts/run_graph.py path/to/document.pdf
    python scripts/run_graph.py path/to/document.pdf --verbose
    python scripts/run_graph.py path/to/document.pdf --json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import structlog

from config import AppConfig, setup_logging
from graph.builder import build_classification_graph
from graph.state import create_initial_state
from pipeline.filesystem import detect_file_type, generate_document_id
from pipeline.graph_adapter import build_logprob_summary, graph_state_to_document_result


@click.command()
@click.argument(
    "file-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as raw JSON")
def main(
    file_path: Path,
    verbose: bool,
    as_json: bool,
) -> None:
    """Classify a single document using the LangGraph hierarchical pipeline."""
    config = AppConfig()  # type: ignore[call-arg]

    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})

    setup_logging(config)
    log = structlog.get_logger()

    file_type = detect_file_type(file_path)
    if not file_type:
        click.echo(f"Error: Unsupported file type: {file_path.suffix}")
        sys.exit(1)

    doc_id = generate_document_id(file_path)

    log.info("graph_pipeline_start", file_path=str(file_path), file_type=file_type)
    start_time = time.monotonic()

    # Build graph
    graph = build_classification_graph(config)

    # Create initial state
    initial_state = create_initial_state(
        document_id=doc_id,
        file_path=str(file_path),
        file_type=file_type,
    )

    # Run graph with a unique thread_id
    thread_config = {"configurable": {"thread_id": doc_id}}

    # Invoke the graph
    final_state = graph.invoke(initial_state, config=thread_config)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    # Check if execution was interrupted (HITL)
    if final_state.get("requires_human_review"):
        click.echo("\n⚠️  HUMAN REVIEW REQUIRED")
        click.echo(f"Document: {file_path.name}")
        click.echo(f"Uncertainty Stage: {final_state.get('uncertainty_stage')}")
        click.echo(f"Root: {final_state.get('root_code')} (margin: {final_state.get('root_margin', 0):.3f})")
        click.echo(f"Sub:  {final_state.get('sub_code', 'N/A')} (margin: {final_state.get('sub_margin', 0):.3f})")
        click.echo(f"\nThread ID: {doc_id}")
        click.echo("Use the Streamlit review page to resolve this document.")
        sys.exit(2)

    # Convert to DocumentResult
    result = graph_state_to_document_result(
        state=final_state,
        file_name=file_path.name,
        processing_time_ms=elapsed_ms,
    )

    if as_json:
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(f"\n{'='*50}")
        click.echo("LANGGRAPH CLASSIFICATION RESULT")
        click.echo(f"{'='*50}")
        click.echo(f"File:      {file_path.name}")
        click.echo(f"Doc ID:    {doc_id}")

        page = result.pages[0]
        click.echo(f"Primary:   {page.classification.primary_class.value}")
        click.echo(f"Subcategory: {page.classification.subcategory.value}")

        if result.hospital_name:
            click.echo(f"Hospital:  {result.hospital_name}")

        # Logprob details
        summary = build_logprob_summary(final_state)
        click.echo(f"\n--- Logprob Analysis ---")
        click.echo(f"Root: {summary['root']['code']} | Margin: {summary['root']['margin']:.3f} | Confidence: {summary['root']['confidence_pct']:.1f}%")
        if summary["sub"]["code"]:
            click.echo(f"Sub:  {summary['sub']['code']} | Margin: {summary['sub']['margin']:.3f} | Confidence: {summary['sub']['confidence_pct']:.1f}%")

        click.echo(f"\nExecution Trail: {' → '.join(summary['execution_trail'])}")
        click.echo(f"Time: {elapsed_ms}ms")


if __name__ == "__main__":
    main()
