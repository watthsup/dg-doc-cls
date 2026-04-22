"""CLI entry point for single document classification.

Usage:
    python scripts/run_single.py path/to/document.pdf
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import structlog

from classifier.llm import LLMClassifier
from config import AppConfig, setup_logging
from ocr.engine import create_di_client
from pipeline.document import process_document
from pipeline.filesystem import detect_file_type, generate_document_id
from schemas.models import DocumentInput


@click.command()
@click.argument(
    "file-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--max-pages", type=int, default=None)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as raw JSON")
def main(
    file_path: Path,
    max_pages: int | None,
    verbose: bool,
    as_json: bool,
) -> None:
    """Classify a single document and print the result."""
    config = AppConfig()  # type: ignore[call-arg]

    if max_pages is not None:
        config = config.model_copy(update={"max_pages": max_pages})
    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})

    setup_logging(config)
    log = structlog.get_logger()

    file_type = detect_file_type(file_path)
    if file_type is None:
        log.error("unsupported_file_type", file_path=str(file_path))
        click.echo(f"Error: Unsupported file type for {file_path.name}")
        sys.exit(1)

    doc_input = DocumentInput(
        document_id=generate_document_id(file_path),
        file_path=file_path.resolve(),
        file_type=file_type,  # type: ignore[arg-type]
    )

    if not as_json:
        click.echo(f"Processing: {file_path.name} ({file_type})")

    # Initialize components
    classifier = LLMClassifier(config)
    di_client = create_di_client(config)

    # Run pipeline
    try:
        result = asyncio.run(
            process_document(
                document=doc_input,
                classifier=classifier,
                config=config,
                di_client=di_client,
            )
        )
    except Exception as e:
        log.exception("processing_failed", error=str(e))
        click.echo(f"Error processing document: {e}")
        sys.exit(1)

    if as_json:
        # Print pure JSON for downstream piping
        print(result.model_dump_json(indent=2))
    else:
        # Pretty print for humans
        click.echo(f"\n{'='*50}")
        click.echo("DOCUMENT SUMMARY")
        click.echo(f"{'='*50}")
        click.echo(f"Document ID: {result.document_id}")
        click.echo(f"Doc Type:    {result.filename_doc_type or 'N/A'}")
        
        click.echo(f"\nProcessing Details:")
        click.echo(f"  - Total Pages: {result.processing_metadata.total_pages}")
        click.echo(f"  - Time:        {result.processing_metadata.processing_time_ms}ms")

        click.echo(f"\n{'='*50}")
        click.echo("PER-PAGE BREAKDOWN")
        click.echo(f"{'='*50}")
        
        for page in result.pages:
            click.echo(f"\n[ Page {page.page_index} ]")
            click.echo(f"  Primary:     {page.classification.primary_class.value}")
            click.echo(f"  Subcategory: {page.classification.subcategory.value}")
            if page.classification.hospital_name:
                click.echo(f"  Hospital:    {page.classification.hospital_name}")
            click.echo(f"  Confidence:  {page.confidence:.3f}")
            
            click.echo(f"  Signals:")
            click.echo(f"    - OCR:      {page.signals.ocr_confidence:.3f}")
            click.echo(f"    - Quality:  {page.signals.quality_score:.3f}")

            if page.quality_assessment.issues:
                click.echo(f"  Issues:")
                for issue in page.quality_assessment.issues:
                    click.echo(f"    ⚠️  {issue}")
        click.echo(f"{'='*50}\n")


if __name__ == "__main__":
    main()
