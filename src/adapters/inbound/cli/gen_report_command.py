from __future__ import annotations

import asyncio
import csv
import pathlib
import sys
from typing import Any

import structlog

from src.infrastructure.config.settings import AppConfig
from src.infrastructure.telemetry.logging import setup_logging
from src.application.use_cases.process_document import DocumentProcessor
from src.application.use_cases.filesystem import scan_documents
from src.adapters.presenters.html_report_presenter import HtmlReportPresenter

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
    
    semaphore = asyncio.Semaphore(concurrency)
    
    jsonl_file = output_dir / "results.jsonl"
    csv_file = output_dir / "results.csv"
    
    # Initialize CSV header
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "page_index", "root_code", "sub_code",
            "root_score", "root_margin", "root_conf_pct",
            "sub_score", "sub_margin", "sub_conf_pct",
            "is_uncertain", "processing_time_ms", 
            "ocr_latency_ms", "root_latency_ms", "sub_latency_ms",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "trail", "ocr_text"
        ])
    
    async def _process_one(f_path: pathlib.Path) -> None:
        async with semaphore:
            try:
                result = await processor.process_file(f_path)
                html = presenter.generate_report(result, str(f_path))
                
                report_path = output_dir / f"{f_path.stem}.html"
                report_path.write_text(html, encoding="utf-8")
                
                # Save JSON result
                with open(jsonl_file, "a", encoding="utf-8") as f:
                    f.write(result.model_dump_json() + "\n")
                
                # Append to CSV
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for p in result.pages:
                        metrics = p.node_metrics or {}
                        p_tokens = sum(m.get("prompt_tokens", 0) for m in metrics.values())
                        c_tokens = sum(m.get("completion_tokens", 0) for m in metrics.values())
                        t_tokens = sum(m.get("total_tokens", 0) for m in metrics.values())
                        root_lat = metrics.get("root_router", {}).get("latency_ms", 0)
                        
                        sub_lat = 0
                        for k, v in metrics.items():
                            if "specialist" in k:
                                sub_lat += v.get("latency_ms", 0)

                        writer.writerow([
                            result.file_name, p.page_index + 1, p.root_code, p.sub_code,
                            f"{p.root_score:.4f}", f"{p.root_margin:.4f}", f"{p.root_confidence_pct:.1f}",
                            f"{p.sub_score:.4f}", f"{p.sub_margin:.4f}", f"{p.sub_confidence_pct:.1f}",
                            p.is_uncertain, result.processing_time_ms,
                            result.pipeline_metrics.get("azure_di_ocr_latency_ms", 0),
                            root_lat, sub_lat,
                            p_tokens, c_tokens, t_tokens,
                            " -> ".join(p.execution_trail), p.ocr_text
                        ])
                
                print(f"  ✅ {f_path.name} -> {report_path.name}")
            except Exception as e:
                print(f"  ❌ {f_path.name}: {e}")

    tasks = [_process_one(fp) for fp in input_files]
    await asyncio.gather(*tasks)
    print(f"\nReport generation complete. Files saved to {output_dir}")
