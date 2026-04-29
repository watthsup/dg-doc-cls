"""CLI entry point for batch document classification.

Processes all documents in a directory, classifying each page independently.

Usage:
    python scripts/run_batch.py --input-dir ./documents --output-dir ./results
    python scripts/run_batch.py --input-dir ./documents --output-dir ./results --json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import click
import structlog

from config import AppConfig, setup_logging
from graph.builder import build_classification_graph
from pipeline.document import process_document_pages
from pipeline.filesystem import scan_documents
from schemas.multi_page import MultiPageResult


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing document files (PDF/images)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Directory for output files (default: ./output)",
)
@click.option("--max-concurrency", type=int, default=3, help="Max documents to process concurrently")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSONL")
def main(
    input_dir: Path,
    output_dir: Path,
    max_concurrency: int,
    verbose: bool,
    as_json: bool,
) -> None:
    """Classify all documents in a directory using the LangGraph pipeline."""
    config = AppConfig()  # type: ignore[call-arg]

    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})

    setup_logging(config)
    log = structlog.get_logger()

    log.info("scanning_directory", input_dir=str(input_dir))
    documents = scan_documents(input_dir)

    if not documents:
        log.warning("no_documents_found", input_dir=str(input_dir))
        click.echo("No supported documents found.")
        sys.exit(0)

    click.echo(f"Found {len(documents)} documents. Processing...")

    asyncio.run(_run_batch(documents, config, output_dir, max_concurrency, as_json))


async def _run_batch(
    documents: list,
    config: AppConfig,
    output_dir: Path,
    max_concurrency: int,
    as_json: bool,
) -> None:
    """Run batch processing with controlled concurrency."""
    log = structlog.get_logger()
    start_time = time.monotonic()

    # Build graph once, reuse for all documents
    graph = build_classification_graph(config, use_checkpointer=False)

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[MultiPageResult] = []
    errors: list[dict] = []

    async def _process_one(doc) -> None:
        async with semaphore:
            try:
                result = await process_document_pages(
                    file_path=doc.file_path,
                    config=config,
                    graph=graph,
                )
                results.append(result)
                click.echo(f"  ✅ {result.file_name}: {result.summary}")
            except Exception as e:
                error_info = {
                    "document_id": doc.document_id,
                    "file_name": doc.file_path.name,
                    "error": str(e),
                }
                errors.append(error_info)
                log.error("document_failed", **error_info)
                click.echo(f"  ❌ {doc.file_path.name}: {e}")

    tasks = [_process_one(doc) for doc in documents]
    await asyncio.gather(*tasks)

    elapsed_s = time.monotonic() - start_time

    # --- Output ---
    output_dir.mkdir(parents=True, exist_ok=True)

    if as_json:
        output_file = output_dir / "batch_results.jsonl"
        with open(output_file, "w") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")
        click.echo(f"\nResults written to: {output_file}")

    if errors:
        error_file = output_dir / "errors.jsonl"
        with open(error_file, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        click.echo(f"Errors written to: {error_file}")

    click.echo(f"\n{'='*50}")
    click.echo("BATCH PROCESSING COMPLETE")
    click.echo(f"{'='*50}")
    click.echo(f"Total:      {len(documents)}")
    click.echo(f"Successful: {len(results)}")
    click.echo(f"Failed:     {len(errors)}")
    click.echo(f"Time:       {elapsed_s:.1f}s")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
