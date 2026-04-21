"""Filesystem scanner — finds document files in a directory."""

from __future__ import annotations

import hashlib
from pathlib import Path

from schemas.models import DocumentInput

_PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
})
_ALL_EXTENSIONS: frozenset[str] = _PDF_EXTENSIONS | _IMAGE_EXTENSIONS


def detect_file_type(path: Path) -> str | None:
    """Detect PDF or image from extension. None = unsupported."""
    suffix = path.suffix.lower()
    if suffix in _PDF_EXTENSIONS:
        return "pdf"
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    return None


def generate_document_id(path: Path) -> str:
    """Filename + short hash for human-readable unique IDs."""
    path_hash = hashlib.md5(  # noqa: S324
        str(path.resolve()).encode()
    ).hexdigest()[:8]
    return f"{path.stem}_{path_hash}"


def scan_documents(input_dir: Path) -> list[DocumentInput]:
    """Scan directory for supported documents. Non-recursive, sorted.

    Raises:
        FileNotFoundError: If input_dir does not exist.
        NotADirectoryError: If path is not a directory.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {input_dir}")

    documents: list[DocumentInput] = []
    for file_path in sorted(input_dir.iterdir()):
        if not file_path.is_file():
            continue
        file_type = detect_file_type(file_path)
        if file_type is None:
            continue
        documents.append(
            DocumentInput(
                document_id=generate_document_id(file_path),
                file_path=file_path.resolve(),
                file_type=file_type,  # type: ignore[arg-type]
            )
        )
    return documents
