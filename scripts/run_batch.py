"""CLI entry point for batch document classification.

Usage:
    python scripts/run_batch.py --input-dir ./documents --output-dir ./results
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import structlog

from config import AppConfig, setup_logging
from exporters.writer import export_batch_result
from pipeline.batch import process_batch
from pipeline.filesystem import scan_documents


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
@click.option(
    "--output-format",
    type=click.Choice(["jsonl", "csv", "both"]),
    default="jsonl",
)
@click.option("--max-concurrency", type=int, default=None)
@click.option("--max-pages", type=int, default=None)
@click.option("--verbose", is_flag=True, default=False)
def main(
    input_dir: Path,
    output_dir: Path,
    output_format: str,
    max_concurrency: int | None,
    max_pages: int | None,
    verbose: bool,
) -> None:
    """Classify documents in a folder and export results."""
    config = AppConfig()  # type: ignore[call-arg]

    if max_concurrency is not None:
        config = config.model_copy(update={"max_concurrency": max_concurrency})
    if max_pages is not None:
        config = config.model_copy(update={"max_pages": max_pages})
    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})
    config = config.model_copy(update={"output_format": output_format})

    setup_logging(config)
    log = structlog.get_logger()

    log.info("scanning_directory", input_dir=str(input_dir))
    documents = scan_documents(input_dir)

    if not documents:
        log.warning("no_documents_found", input_dir=str(input_dir))
        click.echo("No supported documents found.")
        sys.exit(0)

    log.info("documents_found", count=len(documents))
    click.echo(f"Found {len(documents)} documents. Processing...")

    batch_result = asyncio.run(process_batch(documents, config))
    output_files = export_batch_result(batch_result, output_dir, output_format)

    click.echo(f"\n{'='*50}")
    click.echo("BATCH PROCESSING COMPLETE")
    click.echo(f"{'='*50}")
    click.echo(f"Total:      {batch_result.total_documents}")
    click.echo(f"Successful: {batch_result.successful}")
    click.echo(f"Failed:     {batch_result.failed}")
    click.echo("\nOutput files:")
    for f in output_files:
        click.echo(f"  → {f}")

    if batch_result.failed > 0:
        click.echo(
            f"\n⚠️  {batch_result.failed} documents failed. "
            "Check errors.jsonl for details."
        )

    sys.exit(1 if batch_result.failed > 0 else 0)


if __name__ == "__main__":
    main()
