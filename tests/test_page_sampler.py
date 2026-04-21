"""Tests for page_sampler."""

import pytest

from ocr.page_sampler import sample_pages


class TestSamplePages:

    def test_single_page_document(self) -> None:
        assert sample_pages(total_pages=1, max_pages=2) == [0]

    def test_two_page_document(self) -> None:
        assert sample_pages(total_pages=2, max_pages=2) == [0, 1]

    def test_multi_page_document_capped(self) -> None:
        assert sample_pages(total_pages=50, max_pages=2) == [0, 1]

    def test_custom_max_pages(self) -> None:
        assert sample_pages(total_pages=10, max_pages=5) == [0, 1, 2, 3, 4]

    def test_max_pages_one(self) -> None:
        assert sample_pages(total_pages=10, max_pages=1) == [0]

    def test_total_pages_equals_max(self) -> None:
        assert sample_pages(total_pages=3, max_pages=3) == [0, 1, 2]

    def test_invalid_total_pages_zero(self) -> None:
        with pytest.raises(ValueError, match="total_pages must be >= 1"):
            sample_pages(total_pages=0)

    def test_invalid_total_pages_negative(self) -> None:
        with pytest.raises(ValueError, match="total_pages must be >= 1"):
            sample_pages(total_pages=-1)

    def test_invalid_max_pages_zero(self) -> None:
        with pytest.raises(ValueError, match="max_pages must be >= 1"):
            sample_pages(total_pages=5, max_pages=0)

    def test_returns_list_type(self) -> None:
        result = sample_pages(total_pages=5, max_pages=2)
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
