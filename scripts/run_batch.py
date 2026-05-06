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
def main(
    input_dir: Path,
    output_dir: Path,
    max_concurrency: int,
    verbose: bool,
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

    asyncio.run(_run_batch(documents, config, output_dir, max_concurrency))


async def _run_batch(
    documents: list,
    config: AppConfig,
    output_dir: Path,
    max_concurrency: int,
) -> None:
    """Run batch processing with controlled concurrency."""
    log = structlog.get_logger()
    start_time = time.monotonic()

    # --- 1. Setup Runner ---
    from graph.builder import ClassificationRunner
    runner = ClassificationRunner.from_env()

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_dir / "batch_results.jsonl"
    csv_file = output_dir / "batch_results.csv"
    error_file = output_dir / "errors.jsonl"

    # Initialize CSV header
    import csv
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "page_index", "root_code", "sub_code",
            "root_score", "root_margin", "root_conf_pct",
            "sub_score", "sub_margin", "sub_conf_pct",
            "hospital_name", "is_uncertain", "processing_time_ms", 
            "ocr_latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
            "node_metrics", "trail", "ocr_text"
        ])

    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    results_count = 0
    errors_count = 0

    async def _process_one(doc) -> None:
        nonlocal results_count, errors_count
        async with semaphore:
            try:
                state = await runner.run(str(doc.file_path))
                if "error" in state:
                    raise RuntimeError(state["error"])
                result = state["final_result"]
                
                # Incremental write to files
                async with write_lock:
                    # 1. Append to JSONL
                    with open(jsonl_file, "a", encoding="utf-8") as f:
                        f.write(result.model_dump_json() + "\n")
                    
                    # 2. Append to CSV
                    with open(csv_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for p in result.pages:
                            # Calculate total tokens for this page
                            metrics = p.node_metrics or {}
                            p_tokens = sum(m.get("prompt_tokens", 0) for m in metrics.values())
                            c_tokens = sum(m.get("completion_tokens", 0) for m in metrics.values())
                            t_tokens = sum(m.get("total_tokens", 0) for m in metrics.values())

                            writer.writerow([
                                result.file_name, p.page_index + 1, p.root_code, p.sub_code,
                                f"{p.root_score:.4f}", f"{p.root_margin:.4f}", f"{p.root_confidence_pct:.1f}",
                                f"{p.sub_score:.4f}", f"{p.sub_margin:.4f}", f"{p.sub_confidence_pct:.1f}",
                                p.hospital_name or "", p.is_uncertain, result.processing_time_ms,
                                result.pipeline_metrics.get("azure_di_ocr_latency_ms", 0),
                                p_tokens, c_tokens, t_tokens,
                                json.dumps(metrics), " -> ".join(p.execution_trail), p.ocr_text
                            ])
                    results_count += 1

                click.echo(f"  ✅ {result.file_name}: {result.summary}")

            except Exception as e:
                error_info = {
                    "document_id": doc.document_id,
                    "file_name": doc.file_path.name,
                    "error": str(e),
                }
                async with write_lock:
                    with open(error_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_info) + "\n")
                    errors_count += 1
                
                log.error("document_failed", **error_info)
                click.echo(f"  ❌ {doc.file_path.name}: {e}")

    # --- 2. Process documents ---
    tasks = []
    for doc in documents:
        tasks.append(_process_one(doc))
    await asyncio.gather(*tasks)

    elapsed_s = time.monotonic() - start_time

    click.echo(f"\n{'='*50}")
    click.echo("BATCH PROCESSING COMPLETE")
    click.echo(f"{'='*50}")
    click.echo(f"Total:      {len(documents)}")
    click.echo(f"Successful: {results_count}")
    click.echo(f"Failed:     {errors_count}")
    click.echo(f"Time:       {elapsed_s:.1f}s")
    click.echo(f"Results:    {output_dir}")

    if errors_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
