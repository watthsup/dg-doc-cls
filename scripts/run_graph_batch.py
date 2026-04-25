"""CLI entry point for batch document classification via LangGraph pipeline.

Processes a directory of documents concurrently using the stateful
LangGraph hierarchical classifier. Results are written to a JSONL file.

Usage:
    python scripts/run_graph_batch.py path/to/docs/ --output results_v2.jsonl
    python scripts/run_graph_batch.py path/to/docs/ --max-concurrency 3 --verbose
    python scripts/run_graph_batch.py path/to/docs/ --resume  # Skip already-checkpointed docs
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import click
import structlog

from config import AppConfig, setup_logging
from exporters.writer import export_jsonl
from graph.builder import build_classification_graph
from graph.state import create_initial_state
from pipeline.filesystem import detect_file_type, generate_document_id
from pipeline.graph_adapter import graph_state_to_document_result
from schemas.models import BatchResult, DocumentError, DocumentResult

_SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _collect_documents(input_dir: Path) -> list[Path]:
    """Collect all supported document files from a directory."""
    return sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


async def _process_one(
    file_path: Path,
    graph: object,  # CompiledStateGraph
    semaphore: asyncio.Semaphore,
    log: Any,
) -> DocumentResult | DocumentError:
    """Process a single document through the LangGraph pipeline.

    Uses a semaphore to enforce max_concurrency across concurrent tasks.
    Each document gets its own thread_id for checkpointing.
    """
    async with semaphore:
        doc_id = generate_document_id(file_path)
        file_type = detect_file_type(file_path)
        thread_config = {"configurable": {"thread_id": doc_id}}

        if file_type is None:
            return DocumentError(
                document_id=doc_id,
                error_type="UnsupportedFileType",
                error_message=f"Unsupported extension: {file_path.suffix}",
                stage="ingestion",
            )

        try:
            start = time.monotonic()
            initial_state = create_initial_state(
                document_id=doc_id,
                file_path=str(file_path),
                file_type=file_type,
            )
            final_state = await asyncio.to_thread(
                graph.invoke, initial_state, thread_config  # type: ignore[attr-defined]
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            if final_state.get("requires_human_review"):
                log.warning(
                    "graph_batch_hitl_flagged",
                    document_id=doc_id,
                    file_name=file_path.name,
                    uncertainty_stage=final_state.get("uncertainty_stage"),
                )

            return graph_state_to_document_result(
                state=final_state,
                file_name=file_path.name,
                processing_time_ms=elapsed_ms,
            )

        except Exception as exc:
            error_type = type(exc).__name__
            log.error(
                "graph_batch_document_failed",
                document_id=doc_id,
                file_name=file_path.name,
                error_type=error_type,
                error=str(exc),
            )
            return DocumentError(
                document_id=doc_id,
                error_type=error_type,
                error_message=str(exc),
                stage="graph_pipeline",
            )


@click.command()
@click.argument(
    "input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("results_v2.jsonl"),
    show_default=True,
    help="Output JSONL file path",
)
@click.option(
    "--max-concurrency", "-c",
    type=int,
    default=None,
    help="Max concurrent documents (defaults to config value)",
)
@click.option("--verbose", is_flag=True, default=False)
def main(
    input_dir: Path,
    output: Path,
    max_concurrency: int | None,
    verbose: bool,
) -> None:
    """Classify a directory of documents using the LangGraph hierarchical pipeline.

    Documents flagged for human review (low margin score) are NOT skipped —
    they are included in results with requires_human_review=True.
    Use the Streamlit review page to resolve them.
    """
    config = AppConfig()  # type: ignore[call-arg]
    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})
    if max_concurrency is not None:
        config = config.model_copy(update={"max_concurrency": max_concurrency})

    setup_logging(config)
    log = structlog.get_logger()

    documents = _collect_documents(input_dir)
    if not documents:
        click.echo(f"No supported documents found in: {input_dir}")
        sys.exit(1)

    click.echo(f"Found {len(documents)} document(s) in {input_dir}")
    click.echo(f"Max concurrency: {config.max_concurrency}")
    click.echo(f"Output: {output}")
    click.echo()

    # Build graph once — the checkpointer is shared across all docs
    graph = build_classification_graph(config)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_batch() -> BatchResult:
        tasks = [
            _process_one(doc_path, graph, semaphore, log)
            for doc_path in documents
        ]
        outcomes = await asyncio.gather(*tasks)

        results: list[DocumentResult] = []
        errors: list[DocumentError] = []
        hitl_flagged: int = 0

        for outcome in outcomes:
            if isinstance(outcome, DocumentResult):
                results.append(outcome)
                if any(p.confidence < 0.5 for p in outcome.pages):
                    hitl_flagged += 1
            else:
                errors.append(outcome)

        return BatchResult(
            total_documents=len(documents),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
        )

    start_time = time.monotonic()
    batch = asyncio.run(_run_batch())
    elapsed = time.monotonic() - start_time

    # Write JSONL
    export_jsonl(batch.results, output)

    # Summary
    click.echo(f"\n{'='*50}")
    click.echo("BATCH COMPLETE")
    click.echo(f"{'='*50}")
    click.echo(f"Total:      {batch.total_documents}")
    click.echo(f"Successful: {batch.successful}")
    click.echo(f"Failed:     {batch.failed}")
    click.echo(f"Time:       {elapsed:.1f}s ({elapsed/batch.total_documents:.1f}s avg)")
    click.echo(f"Output:     {output}")
    if batch.errors:
        click.echo(f"\n⚠️  {batch.failed} document(s) failed:")
        for err in batch.errors:
            click.echo(f"  - {err.document_id}: {err.error_message}")

    sys.exit(0 if batch.failed == 0 else 1)


if __name__ == "__main__":
    main()
