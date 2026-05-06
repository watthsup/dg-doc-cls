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
from pipeline.document import process_document_pages


@click.command()
@click.argument(
    "file-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as raw JSON")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Save result to this file (JSON)",
)
def main(
    file_path: Path,
    verbose: bool,
    as_json: bool,
    output: Path | None = None,
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
        
        # Show Document-level OCR latency
        ocr_lat = result.pipeline_metrics.get("azure_di_ocr_latency_ms", 0)
        if ocr_lat > 0:
            click.echo(f"OCR Latency: {ocr_lat}ms")

        for page in result.pages:
            click.echo(f"\n--- Page {page.page_index + 1} ---")
            click.echo(f"  Root: {page.root_code} (Score: {page.root_score:.3f}, Margin: {page.root_margin:.3f}, Conf: {page.root_confidence_pct:.1f}%)")
            if page.sub_code:
                click.echo(f"  Sub:  {page.sub_code} (Score: {page.sub_score:.3f}, Margin: {page.sub_margin:.3f}, Conf: {page.sub_confidence_pct:.1f}%)")

            if page.hospital_name:
                click.echo(f"  Hospital:       {page.hospital_name}")

            if page.is_uncertain:
                click.echo(f"  ⚠️  UNCERTAIN — flagged for human review")

            # Display Node Metrics (Latency & Tokens)
            if page.node_metrics:
                click.echo(f"  Metrics:")
                for node, m in page.node_metrics.items():
                    perf = f"{m.get('latency_ms', 0)}ms"
                    tokens = ""
                    if "total_tokens" in m:
                        tokens = f" (Tokens: {m.get('total_tokens')} [In:{m.get('prompt_tokens')}/Out:{m.get('completion_tokens')}])"
                    click.echo(f"    - {node:20}: {perf}{tokens}")

            click.echo(f"  Trail: {' → '.join(page.execution_trail)}")

        click.echo(f"\n{'='*60}")
        click.echo(f"Summary: {result.summary}")

        if result.has_uncertain_pages:
            click.echo("\n⚠️  Some pages were flagged as uncertain. Review recommended.")

    # --- Save to file if requested ---
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(result.model_dump_json(indent=2))
        click.echo(f"\n✅ Result saved to: {output}")

    if result.has_uncertain_pages:
        sys.exit(2)


if __name__ == "__main__":
    main()
