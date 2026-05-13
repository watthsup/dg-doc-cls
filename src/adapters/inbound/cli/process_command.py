from __future__ import annotations

import asyncio
import pathlib
import sys

import structlog

from src.infrastructure.config.settings import AppConfig
from src.infrastructure.telemetry.logging import setup_logging
from src.application.use_cases.process_document import DocumentProcessor
from src.application.use_cases.filesystem import scan_documents
from src.adapters.presenters.batch_summary_exporter import BatchSummaryExporter

log = structlog.get_logger()

def execute_process_command(
    file: str | None,
    directory: str | None,
    output_dir: str,
    graph_type: str,
    max_concurrency: int,
    verbose: bool,
    json_output: bool = False,
) -> int:
    """Implementation of the 'process' CLI command."""
    config = AppConfig()  # type: ignore[call-arg]
    if verbose:
        config = config.model_copy(update={"log_level": "DEBUG"})
    
    setup_logging(config)
    
    # 1. Resolve inputs
    input_files: list[pathlib.Path] = []
    if file:
        f_path = pathlib.Path(file)
        if not f_path.exists():
            print(f"Error: File {file} does not exist", file=sys.stderr)
            return 1
        input_files.append(f_path)
    elif directory:
        d_path = pathlib.Path(directory)
        if not d_path.exists():
            print(f"Error: Directory {directory} does not exist", file=sys.stderr)
            return 1
        input_files = [d.file_path for d in scan_documents(d_path)]
    
    if not input_files:
        print("No supported documents found.", file=sys.stderr)
        return 0

    print(f"Found {len(input_files)} documents. Processing...", file=sys.stderr)
    
    # 2. Setup output
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Run
    asyncio.run(_run_pipeline(input_files, config, out_path, graph_type, max_concurrency, json_output))
    return 0

async def _run_pipeline(
    input_files: list[pathlib.Path],
    config: AppConfig,
    output_dir: pathlib.Path,
    graph_type: str,
    max_concurrency: int,
    json_output: bool = False,
) -> None:
    processor = DocumentProcessor(config=config, graph_type=graph_type)
    exporter = BatchSummaryExporter(output_dir)

    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    results_count = 0
    errors_count = 0

    async def _process_one(f_path: pathlib.Path) -> None:
        nonlocal results_count, errors_count
        async with semaphore:
            try:
                result = await processor.process_file(f_path)
                
                async with write_lock:
                    exporter.append_result(result)
                    results_count += 1
                print(f"  ✅ {result.file_name}: {result.summary}", file=sys.stderr)
                if json_output:
                    print(result.model_dump_json(indent=2), file=sys.stdout)
            except Exception as e:
                async with write_lock:
                    exporter.append_error(f_path.name, str(e))
                    errors_count += 1
                print(f"  ❌ {f_path.name}: {e}", file=sys.stderr)

    tasks = [_process_one(fp) for fp in input_files]
    await asyncio.gather(*tasks)

    print(f"\nProcessing complete. Success: {results_count}, Errors: {errors_count}", file=sys.stderr)
    print(f"Results saved to {output_dir}", file=sys.stderr)
