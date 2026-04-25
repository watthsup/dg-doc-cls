"""Tests for pipeline/graph_adapter.py — GraphState to model conversion."""

from __future__ import annotations

import pytest

from pipeline.graph_adapter import (
    build_logprob_summary,
    graph_state_to_document_result,
    graph_state_to_page_result,
)
from schemas.models import PrimaryClass, QualityAssessment, Subcategory


class TestGraphStateToPageResult:
    """Test conversion from GraphState to PageResult."""

    def test_medical_lab_conversion(self) -> None:
        state = {
            "root_code": "MED",
            "sub_code": "LAB",
            "root_confidence_pct": 99.0,
            "sub_confidence_pct": 95.0,
            "hospital_name": "Test Hospital",
            "azure_ocr_text": "CBC results...",
        }
        result = graph_state_to_page_result(state, page_idx=0, ocr_confidence=0.9)
        assert result.classification.primary_class == PrimaryClass.MEDICAL
        assert result.classification.subcategory == Subcategory.LAB
        assert result.classification.hospital_name == "Test Hospital"
        assert result.ocr_text == "CBC results..."

    def test_nonmedical_passport_conversion(self) -> None:
        state = {
            "root_code": "NON",
            "sub_code": "PAS",
            "root_confidence_pct": 98.0,
            "sub_confidence_pct": 95.0,
            "hospital_name": None,
            "azure_ocr_text": "PASSPORT Republic of Thailand...",
        }
        result = graph_state_to_page_result(state)
        assert result.classification.primary_class == PrimaryClass.NON_MEDICAL
        assert result.classification.subcategory == Subcategory.PASSPORT
        assert result.classification.hospital_name is None

    def test_nonmedical_financial_conversion(self) -> None:
        state = {
            "root_code": "NON",
            "sub_code": "FIN",
            "root_confidence_pct": 98.0,
            "sub_confidence_pct": 90.0,
            "hospital_name": None,
            "azure_ocr_text": "Invoice...",
        }
        result = graph_state_to_page_result(state)
        assert result.classification.primary_class == PrimaryClass.NON_MEDICAL
        assert result.classification.subcategory == Subcategory.FINANCIAL

    def test_oth_maps_contextually_for_medical(self) -> None:
        state = {
            "root_code": "MED",
            "sub_code": "OTH",
            "root_confidence_pct": 50.0,
            "sub_confidence_pct": 50.0,
        }
        result = graph_state_to_page_result(state)
        assert result.classification.primary_class == PrimaryClass.MEDICAL
        assert result.classification.subcategory == Subcategory.MEDICAL_OTHER

    def test_oth_maps_contextually_for_nonmedical(self) -> None:
        state = {
            "root_code": "NON",
            "sub_code": "OTH",
            "root_confidence_pct": 50.0,
            "sub_confidence_pct": 50.0,
        }
        result = graph_state_to_page_result(state)
        assert result.classification.primary_class == PrimaryClass.NON_MEDICAL
        assert result.classification.subcategory == Subcategory.OTHER

    def test_invalid_code_falls_back(self) -> None:
        state = {
            "root_code": "XXX",
            "sub_code": "YYY",
            "root_confidence_pct": 50.0,
            "sub_confidence_pct": 50.0,
        }
        result = graph_state_to_page_result(state)
        # Should fallback to MEDICAL + MEDICAL_OTHER
        assert result.classification.primary_class == PrimaryClass.MEDICAL
        assert result.classification.subcategory == Subcategory.MEDICAL_OTHER

    def test_nonmedical_invalid_sub_falls_back_to_other(self) -> None:
        state = {
            "root_code": "NON",
            "sub_code": "INVALID",
            "root_confidence_pct": 50.0,
            "sub_confidence_pct": 50.0,
        }
        result = graph_state_to_page_result(state)
        assert result.classification.primary_class == PrimaryClass.NON_MEDICAL
        assert result.classification.subcategory == Subcategory.OTHER

    def test_confidence_calculation(self) -> None:
        state = {
            "root_code": "MED",
            "sub_code": "HCK",
            "root_confidence_pct": 80.0,
            "sub_confidence_pct": 60.0,
        }
        result = graph_state_to_page_result(state)
        # (80/100 + 60/100) / 2 = 0.7
        assert result.confidence == pytest.approx(0.7)

    def test_with_quality_assessment(self) -> None:
        qa = QualityAssessment(
            issues=["High blur"],
            skew_angle=2.5,
            blur_score=0.2,
            contrast_score=0.8,
            quality_score=0.5,
        )
        state = {
            "root_code": "MED",
            "sub_code": "LAB",
            "root_confidence_pct": 90.0,
            "sub_confidence_pct": 85.0,
        }
        result = graph_state_to_page_result(state, quality_assessment=qa)
        assert result.quality_assessment.issues == ["High blur"]
        assert result.quality_assessment.skew_angle == 2.5


class TestGraphStateToDocumentResult:
    """Test conversion from GraphState to DocumentResult."""

    def test_basic_conversion(self) -> None:
        state = {
            "document_id": "doc_001",
            "root_code": "MED",
            "sub_code": "LAB",
            "root_confidence_pct": 99.0,
            "sub_confidence_pct": 95.0,
            "hospital_name": "BNH Hospital",
            "azure_ocr_text": "Lab results...",
        }
        result = graph_state_to_document_result(
            state,
            file_name="test.pdf",
            processing_time_ms=1500,
            total_pages=3,
        )
        assert result.document_id == "doc_001"
        assert result.file_name == "test.pdf"
        assert result.hospital_name == "BNH Hospital"
        assert len(result.pages) == 1
        assert result.processing_metadata.processing_time_ms == 1500
        assert result.processing_metadata.total_pages == 3

    def test_backward_compatible_with_existing_exporters(self) -> None:
        """The output DocumentResult should be serializable by existing exporters."""
        import json

        state = {
            "document_id": "compat_test",
            "root_code": "NON",
            "sub_code": "PAS",
            "root_confidence_pct": 95.0,
            "sub_confidence_pct": 92.0,
            "hospital_name": None,
            "azure_ocr_text": "PASSPORT P<THA...",
        }
        result = graph_state_to_document_result(state, file_name="passport.jpg")

        json_str = result.model_dump_json()
        data = json.loads(json_str)
        assert "document_id" in data
        assert "pages" in data
        assert data["pages"][0]["classification"]["primary_class"] == "non_medical"
        assert data["pages"][0]["classification"]["subcategory"] == "passport"


class TestBuildLogprobSummary:
    """Test the logprob summary builder."""

    def test_summary_structure(self) -> None:
        state = {
            "root_code": "MED",
            "sub_code": "LAB",
            "root_margin": 4.5,
            "sub_margin": 2.1,
            "root_confidence_pct": 99.0,
            "sub_confidence_pct": 88.0,
            "root_logprobs": {"top1_token": "MED", "top1_logprob": -0.01},
            "sub_logprobs": {"top1_token": "LAB", "top1_logprob": -0.12},
            "is_uncertain": False,
            "uncertainty_stage": None,
            "requires_human_review": False,
            "execution_trail": ["ocr_ingestion", "root_router", "med_specialist"],
        }
        summary = build_logprob_summary(state)
        assert summary["root"]["code"] == "MED"
        assert summary["sub"]["code"] == "LAB"
        assert summary["root"]["margin"] == 4.5
        assert summary["is_uncertain"] is False
        assert len(summary["execution_trail"]) == 3
