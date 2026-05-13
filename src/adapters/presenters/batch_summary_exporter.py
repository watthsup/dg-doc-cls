"""Batch summary exporter — CSV + JSONL output for performance analysis.

Mirrors the pattern from doc-structure-agent/BatchSummaryExporter,
adapted for the doc-cls classification pipeline.

Both `process` and `gen-report` CLI commands use this single exporter
instead of duplicating CSV/JSONL writing logic inline.
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

from src.domain.models.multi_page import MultiPageResult


class BatchSummaryExporter:
    """Exports batch classification results to CSV and JSONL formats."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.csv_path = output_dir / "batch_results.csv"
        self.jsonl_path = output_dir / "batch_results.jsonl"
        self.error_path = output_dir / "errors.jsonl"
        self._initialized = False

    def _initialize_files(self) -> None:
        if self._initialized:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV with headers
        headers = [
            "file_name", "page_index", "doc_id",
            "root_code", "sub_code",
            "root_score", "root_margin", "root_conf_pct",
            "sub_score", "sub_margin", "sub_conf_pct",
            "is_uncertain",
            "ocr_latency_ms", "root_latency_ms", "sub_latency_ms",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "processing_time_ms", "node_metrics",
            "trail", "error",
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        # Create/clear JSONL files
        with open(self.jsonl_path, "w", encoding="utf-8"):
            pass

        self._initialized = True

    def append_result(self, result: MultiPageResult) -> None:
        """Append a successful classification result to CSV and JSONL."""
        self._initialize_files()

        # 1. Append to JSONL (full result)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

        # 2. Append to CSV (page-level breakdown)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            ocr_lat = result.pipeline_metrics.get("azure_di_ocr_latency_ms", 0)

            if not result.pages:
                # Write one row even if no page results
                writer.writerow([
                    result.file_name, 1, result.document_id,
                    "n/a", "n/a",
                    "0.0000", "0.0000", "0.0",
                    "0.0000", "0.0000", "0.0",
                    False,
                    f"{ocr_lat:.0f}", 0, 0,
                    0, 0, 0,
                    result.processing_time_ms, "{}",
                    "", "",
                ])
                return

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

                # Build node_metrics JSON matching doc-structure-agent format
                node_metrics_obj: dict[str, Any] = {
                    "ocr": {"latency_ms": ocr_lat},
                    "root_router": metrics.get("root_router", {}),
                }
                # Include whichever specialist ran
                for k in metrics:
                    if "specialist" in k:
                        node_metrics_obj[k] = metrics[k]

                writer.writerow([
                    result.file_name, p.page_index + 1, result.document_id,
                    p.root_code, p.sub_code,
                    f"{p.root_score:.4f}", f"{p.root_margin:.4f}", f"{p.root_confidence_pct:.1f}",
                    f"{p.sub_score:.4f}", f"{p.sub_margin:.4f}", f"{p.sub_confidence_pct:.1f}",
                    p.is_uncertain,
                    f"{ocr_lat:.0f}", root_lat, sub_lat,
                    p_tokens, c_tokens, t_tokens,
                    result.processing_time_ms, json.dumps(node_metrics_obj),
                    " -> ".join(p.execution_trail), "",
                ])

    def append_error(self, filename: str, error: str) -> None:
        """Append a processing error to the error log."""
        self._initialize_files()

        error_info = {"file": filename, "error": error}
        with open(self.error_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_info, ensure_ascii=False) + "\n")
