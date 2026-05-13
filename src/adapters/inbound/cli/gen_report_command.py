from __future__ import annotations

import asyncio
import pathlib
import sys

import structlog

from src.infrastructure.config.settings import AppConfig
from src.infrastructure.telemetry.logging import setup_logging
from src.application.use_cases.process_document import DocumentProcessor
from src.application.use_cases.filesystem import scan_documents
from src.adapters.presenters.html_report_presenter import HtmlReportPresenter
from src.adapters.presenters.batch_summary_exporter import BatchSummaryExporter

log = structlog.get_logger()

def execute_gen_report_command(
    directory: str,
    output_dir: str,
    graph_type: str,
    concurrency: int,
    verbose: bool,
) -> int:
    """Implementation of the 'gen-report' CLI command."""
    config = AppConfig()  # type: ignore[call-arg]
    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})
    
    setup_logging(config)
    
    d_path = pathlib.Path(directory)
    if not d_path.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return 1
        
    input_files = scan_documents(d_path)
    if not input_files:
        print("No supported documents found.", file=sys.stderr)
        return 0

    print(f"Generating reports for {len(input_files)} documents...", file=sys.stderr)
    
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = [doc.file_path for doc in input_files]
    asyncio.run(_run_gen_report(file_paths, config, out_path, graph_type, concurrency))
    return 0

async def _run_gen_report(
    input_files: list[pathlib.Path],
    config: AppConfig,
    output_dir: pathlib.Path,
    graph_type: str,
    concurrency: int,
) -> None:
    processor = DocumentProcessor(config=config, graph_type=graph_type)
    presenter = HtmlReportPresenter()
    exporter = BatchSummaryExporter(output_dir)
    
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    results_count = 0
    errors_count = 0
    
    async def _process_one(f_path: pathlib.Path) -> None:
        nonlocal results_count, errors_count
        async with semaphore:
            try:
                result = await processor.process_file(f_path)
                html = presenter.generate_report(result, str(f_path))
                
                async with write_lock:
                    # Save HTML report
                    report_path = output_dir / f"{f_path.stem}.html"
                    report_path.write_text(html, encoding="utf-8")
                    
                    # Save batch summary (CSV + JSONL)
                    exporter.append_result(result)
                    results_count += 1
                
                print(f"  ✅ {f_path.name} -> {f_path.stem}.html")
            except Exception as e:
                async with write_lock:
                    exporter.append_error(f_path.name, str(e))
                    errors_count += 1
                print(f"  ❌ {f_path.name}: {e}")

    tasks = [_process_one(fp) for fp in input_files]
    await asyncio.gather(*tasks)
    
    print(f"\nReport generation complete. Success: {results_count}, Errors: {errors_count}")
    print(f"Files saved to {output_dir}")
