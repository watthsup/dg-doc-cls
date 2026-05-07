"""Page sampling — selects which pages to process from a document.

Default: first N pages. Extensible for future text-density strategies.
"""

from __future__ import annotations


def sample_pages(total_pages: int, max_pages: int = 2) -> list[int]:
    """Select page indices to process.

    Takes first `max_pages` pages because document headers and key
    identifiers are overwhelmingly on pages 1-2.

    Args:
        total_pages: Total number of pages in the document.
        max_pages: Maximum pages to select.

    Returns:
        List of zero-indexed page indices.

    Raises:
        ValueError: If total_pages < 1 or max_pages < 1.
    """
    if total_pages < 1:
        raise ValueError(f"total_pages must be >= 1, got {total_pages}")
    if max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")

    return list(range(min(total_pages, max_pages)))
