"""CLI entry point for single document classification via LangGraph pipeline.

Supports multi-page documents — each page is classified independently.

Usage:
    python scripts/run_graph.py path/to/document.pdf
    python scripts/run_graph.py path/to/document.pdf --verbose
    python scripts/run_graph.py path/to/document.pdf --json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import structlog

from config import AppConfig, setup_logging
from pipeline.document_processor import process_document_pages


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
    """Classify a document using the LangGraph hierarchical pipeline.

    Multi-page documents (PDF/TIF) are split and each page is classified independently.
    """
    config = AppConfig()  # type: ignore[call-arg]

    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})

    setup_logging(config)

    result = asyncio.run(process_document_pages(file_path, config=config))

    if as_json:
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(f"\n{'='*60}")
        click.echo("LANGGRAPH CLASSIFICATION RESULT")
        click.echo(f"{'='*60}")
        click.echo(f"File:        {result.file_name}")
        click.echo(f"Doc ID:      {result.document_id}")
        click.echo(f"Total Pages: {result.total_pages}")
        click.echo(f"Time:        {result.processing_time_ms}ms")

        for page in result.pages:
            click.echo(f"\n--- Page {page.page_index + 1} ---")
            click.echo(f"  Classification: {page.root_code} → {page.sub_code}")
            click.echo(f"  Root Margin:    {page.root_margin:.3f} (Confidence: {page.root_confidence_pct:.1f}%)")
            click.echo(f"  Sub Margin:     {page.sub_margin:.3f} (Confidence: {page.sub_confidence_pct:.1f}%)")

            if page.hospital_name:
                click.echo(f"  Hospital:       {page.hospital_name}")

            if page.is_uncertain:
                click.echo(f"  ⚠️  UNCERTAIN — flagged for human review")

            click.echo(f"  Trail: {' → '.join(page.execution_trail)}")

        click.echo(f"\n{'='*60}")
        click.echo(f"Summary: {result.summary}")

        if result.has_uncertain_pages:
            click.echo("\n⚠️  Some pages were flagged as uncertain. Review recommended.")
            sys.exit(2)


if __name__ == "__main__":
    main()
