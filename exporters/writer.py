"""Export adapter — writes results to JSONL and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from schemas.models import BatchResult, DocumentResult


def export_jsonl(results: list[DocumentResult], output_path: Path) -> Path:
    """Export results to JSONL (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")
    return output_path


def export_csv(results: list[DocumentResult], output_path: Path) -> Path:
    """Export results to flattened CSV format (one row per page)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["document_id", "file_name"])
        return output_path

    flat_rows = []
    for doc in results:
        doc_dict = doc.model_dump()
        pages = doc_dict.pop("pages", [])
        doc_base = _flatten_dict(doc_dict)
        
        if not pages:
            flat_rows.append(doc_base)
        else:
            for page in pages:
                page_flat = _flatten_dict(page, parent_key="page")
                row = {**doc_base, **page_flat}
                flat_rows.append(row)

    if not flat_rows:
        return output_path

    # Extract all unique fieldnames and sort them
    fieldnames_set = set()
    for row in flat_rows:
        fieldnames_set.update(row.keys())
    fieldnames = sorted(list(fieldnames_set))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    return output_path


def _flatten_dict(
    d: dict[str, object],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, object]:
    """Recursively flatten nested dict with dot-separated keys."""
    items: list[tuple[str, object]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            items.append((new_key, json.dumps(value, ensure_ascii=False)))
        else:
            items.append((new_key, value))
    return dict(items)


def export_batch_result(
    batch_result: BatchResult,
    output_dir: Path,
    output_format: str = "jsonl",
) -> list[Path]:
    """Export batch result to specified format(s)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []

    if output_format in ("jsonl", "both"):
        created_files.append(
            export_jsonl(batch_result.results, output_dir / "results.jsonl")
        )

    if output_format in ("csv", "both"):
        created_files.append(
            export_csv(batch_result.results, output_dir / "results.csv")
        )

    if batch_result.errors:
        errors_path = output_dir / "errors.jsonl"
        with open(errors_path, "w", encoding="utf-8") as f:
            for error in batch_result.errors:
                f.write(error.model_dump_json() + "\n")
        created_files.append(errors_path)

    return created_files
