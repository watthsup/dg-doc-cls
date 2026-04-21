"""Unit tests for schemas/models.py.

Tests parse from JSON/dict — the same entry points used in production:
- LLM response → LLMOutput.model_validate_json()
- Azure DI response → OCR models
- JSONL file read → DocumentResult.model_validate_json()
- Round-trips: object → model_dump_json() → model_validate_json()
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from schemas.models import (
    BatchResult,
    ClassificationResult,
    DocumentError,
    DocumentInput,
    DocumentResult,
    LLMOutput,
    OCRPageResult,
    OCRResult,
    OCRWordResult,
    PrimaryClass,
    ProcessingMetadata,
    QualityAssessment,
    SignalScores,
    Subcategory,
    VALID_SUBCATEGORIES,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestPrimaryClass:

    def test_values(self) -> None:
        assert PrimaryClass.MEDICAL == "medical"
        assert PrimaryClass.NON_MEDICAL == "non_medical"

    def test_str_enum_is_string(self) -> None:
        assert isinstance(PrimaryClass.MEDICAL, str)

    def test_all_members(self) -> None:
        assert {m.value for m in PrimaryClass} == {"medical", "non_medical"}

    def test_parse_from_string(self) -> None:
        assert PrimaryClass("medical") == PrimaryClass.MEDICAL
        assert PrimaryClass("non_medical") == PrimaryClass.NON_MEDICAL

    def test_invalid_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrimaryClass("unknown")


class TestSubcategory:

    def test_values(self) -> None:
        assert Subcategory.LAB == "lab"
        assert Subcategory.HEALTH_CHECK == "health_check"
        assert Subcategory.MEDICAL_OTHER == "medical_other"
        assert Subcategory.ID == "id"
        assert Subcategory.FINANCIAL == "financial"
        assert Subcategory.OTHER == "other"

    def test_six_subcategories(self) -> None:
        assert len(list(Subcategory)) == 6

    def test_parse_from_string(self) -> None:
        assert Subcategory("lab") == Subcategory.LAB
        assert Subcategory("financial") == Subcategory.FINANCIAL


class TestValidSubcategories:

    def test_medical_subcategories(self) -> None:
        medical = VALID_SUBCATEGORIES[PrimaryClass.MEDICAL]
        assert medical == {Subcategory.LAB, Subcategory.HEALTH_CHECK, Subcategory.MEDICAL_OTHER}

    def test_non_medical_subcategories(self) -> None:
        non_medical = VALID_SUBCATEGORIES[PrimaryClass.NON_MEDICAL]
        assert non_medical == {Subcategory.ID, Subcategory.FINANCIAL, Subcategory.OTHER}

    def test_no_overlap_between_classes(self) -> None:
        assert VALID_SUBCATEGORIES[PrimaryClass.MEDICAL].isdisjoint(
            VALID_SUBCATEGORIES[PrimaryClass.NON_MEDICAL]
        )

    def test_all_subcategories_covered(self) -> None:
        all_in_map = (
            VALID_SUBCATEGORIES[PrimaryClass.MEDICAL]
            | VALID_SUBCATEGORIES[PrimaryClass.NON_MEDICAL]
        )
        assert all_in_map == set(Subcategory)


# ---------------------------------------------------------------------------
# LLMOutput — simulates parsing from LLM JSON response
# ---------------------------------------------------------------------------


class TestLLMOutput:

    def test_parse_from_dict_medical(self) -> None:
        """Simulate LLM structured output response as dict."""
        data = {
            "primary_class": "medical",
            "subcategory": "lab",
            "hospital_name": "โรงพยาบาลกรุงเทพ",
        }
        out = LLMOutput.model_validate(data)
        assert out.primary_class == PrimaryClass.MEDICAL
        assert out.subcategory == Subcategory.LAB
        assert out.hospital_name == "โรงพยาบาลกรุงเทพ"

    def test_parse_from_json_string(self) -> None:
        """Simulate raw JSON string from LLM tool-calling response."""
        json_str = '{"primary_class": "non_medical", "subcategory": "financial", "hospital_name": null}'
        out = LLMOutput.model_validate_json(json_str)
        assert out.primary_class == PrimaryClass.NON_MEDICAL
        assert out.subcategory == Subcategory.FINANCIAL
        assert out.hospital_name is None

    def test_hospital_name_omitted_defaults_to_none(self) -> None:
        """LLM may omit optional field — should default to None."""
        data = {"primary_class": "non_medical", "subcategory": "other"}
        out = LLMOutput.model_validate(data)
        assert out.hospital_name is None

    def test_invalid_primary_class_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMOutput.model_validate({"primary_class": "garbage", "subcategory": "lab"})

    def test_invalid_subcategory_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMOutput.model_validate({"primary_class": "medical", "subcategory": "xyz"})

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMOutput.model_validate({"primary_class": "medical"})  # subcategory missing

    def test_round_trip(self) -> None:
        """Serialize and re-parse — should be identical."""
        original = LLMOutput.model_validate(
            {"primary_class": "medical", "subcategory": "health_check", "hospital_name": "BNH"}
        )
        restored = LLMOutput.model_validate_json(original.model_dump_json())
        assert original == restored


# ---------------------------------------------------------------------------
# OCR models — simulate parsing Azure DI response structures
# ---------------------------------------------------------------------------


class TestOCRWordResult:

    def test_parse_from_dict(self) -> None:
        word = OCRWordResult.model_validate({"text": "glucose", "confidence": 95.0})
        assert word.text == "glucose"
        assert word.confidence == 95.0

    def test_confidence_upper_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OCRWordResult.model_validate({"text": "x", "confidence": 101.0})

    def test_confidence_lower_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OCRWordResult.model_validate({"text": "x", "confidence": -1.0})

    def test_missing_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OCRWordResult.model_validate({"confidence": 80.0})


class TestOCRPageResult:

    def test_parse_from_dict(self) -> None:
        data = {
            "page_index": 0,
            "text": "ผลตรวจเลือด",
            "words": [{"text": "ผลตรวจเลือด", "confidence": 88.5}],
            "mean_confidence": 88.5,
        }
        page = OCRPageResult.model_validate(data)
        assert page.page_index == 0
        assert len(page.words) == 1
        assert page.words[0].text == "ผลตรวจเลือด"

    def test_empty_words_allowed(self) -> None:
        data = {"page_index": 0, "text": "", "words": [], "mean_confidence": 0.0}
        page = OCRPageResult.model_validate(data)
        assert page.words == []

    def test_negative_page_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OCRPageResult.model_validate(
                {"page_index": -1, "text": "", "words": [], "mean_confidence": 0.0}
            )

    def test_mean_confidence_above_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OCRPageResult.model_validate(
                {"page_index": 0, "text": "", "words": [], "mean_confidence": 101.0}
            )


class TestOCRResult:

    def test_parse_from_dict(self) -> None:
        data = {
            "pages": [],
            "merged_text": "Lab result: glucose 5.2",
            "overall_confidence": 0.91,
        }
        result = OCRResult.model_validate(data)
        assert result.overall_confidence == 0.91
        assert result.merged_text == "Lab result: glucose 5.2"

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OCRResult.model_validate(
                {"pages": [], "merged_text": "", "overall_confidence": 1.5}
            )

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OCRResult.model_validate(
                {"pages": [], "merged_text": "", "overall_confidence": -0.1}
            )

    def test_default_pages_and_text(self) -> None:
        result = OCRResult.model_validate({"overall_confidence": 0.0})
        assert result.pages == []
        assert result.merged_text == ""

    def test_nested_page_parsing(self) -> None:
        """Ensure nested OCRPageResult/OCRWordResult are parsed correctly."""
        data = {
            "pages": [
                {
                    "page_index": 0,
                    "text": "Blood Test",
                    "words": [{"text": "Blood", "confidence": 90.0}, {"text": "Test", "confidence": 85.0}],
                    "mean_confidence": 87.5,
                }
            ],
            "merged_text": "Blood Test",
            "overall_confidence": 0.875,
        }
        result = OCRResult.model_validate(data)
        assert len(result.pages) == 1
        assert len(result.pages[0].words) == 2
        assert result.pages[0].words[0].text == "Blood"


# ---------------------------------------------------------------------------
# SignalScores
# ---------------------------------------------------------------------------


class TestSignalScores:

    def test_parse_from_dict(self) -> None:
        data = {"ocr_confidence": 0.8, "keyword_score": 0.6, "quality_score": 0.9}
        signals = SignalScores.model_validate(data)
        assert signals.ocr_confidence == 0.8

    def test_parse_from_json_string(self) -> None:
        json_str = '{"ocr_confidence": 0.75, "keyword_score": 0.5, "quality_score": 0.85}'
        signals = SignalScores.model_validate_json(json_str)
        assert signals.keyword_score == 0.5

    def test_value_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SignalScores.model_validate(
                {"ocr_confidence": 1.1, "keyword_score": 0.5, "quality_score": 0.5}
            )

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SignalScores.model_validate(
                {"ocr_confidence": 0.5, "keyword_score": -0.1, "quality_score": 0.5}
            )


# ---------------------------------------------------------------------------
# QualityAssessment — critical: internal fields must not appear in JSON
# ---------------------------------------------------------------------------


_QUALITY_DICT = {
    "blur_score": 0.8,
    "skew_angle": 1.5,
    "contrast_score": 0.7,
    "issues": [],
    "quality_score": 0.75,
}


class TestQualityAssessment:

    def test_parse_from_dict(self) -> None:
        q = QualityAssessment.model_validate(_QUALITY_DICT)
        assert q.skew_angle == 1.5
        assert q.blur_score == 0.8

    def test_internal_fields_excluded_from_json(self) -> None:
        """blur_score, contrast_score, quality_score must NOT appear in JSON output."""
        q = QualityAssessment.model_validate(_QUALITY_DICT)
        data = json.loads(q.model_dump_json())
        assert "blur_score" not in data
        assert "contrast_score" not in data
        assert "quality_score" not in data

    def test_public_fields_present_in_json(self) -> None:
        q = QualityAssessment.model_validate({**_QUALITY_DICT, "issues": ["High blur"]})
        data = json.loads(q.model_dump_json())
        assert set(data.keys()) == {"issues", "skew_angle"}
        assert data["issues"] == ["High blur"]

    def test_invalid_blur_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QualityAssessment.model_validate({**_QUALITY_DICT, "blur_score": 1.5})

    def test_invalid_quality_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QualityAssessment.model_validate({**_QUALITY_DICT, "quality_score": -0.1})


# ---------------------------------------------------------------------------
# DocumentResult — simulate reading a line from JSONL output
# ---------------------------------------------------------------------------

_RESULT_JSON = """{
    "document_id": "report_abc123ef",
    "classification": {
        "primary_class": "medical",
        "subcategory": "lab"
    },
    "confidence": 0.82,
    "signals": {
        "ocr_confidence": 0.75,
        "keyword_score": 0.80,
        "quality_score": 0.90
    },
    "quality_assessment": {
        "issues": [],
        "skew_angle": 1.2
    },
    "processing_metadata": {
        "pages_used": [0, 1],
        "processing_time_ms": 3200
    }
}"""


class TestDocumentResult:

    def test_parse_from_jsonl_line(self) -> None:
        """Simulate reading a line from results.jsonl."""
        result = DocumentResult.model_validate_json(_RESULT_JSON)
        assert result.document_id == "report_abc123ef"
        assert result.classification.primary_class == PrimaryClass.MEDICAL
        assert result.classification.subcategory == Subcategory.LAB
        assert result.confidence == 0.82
        assert result.processing_metadata.pages_used == [0, 1]

    def test_json_output_matches_contract(self) -> None:
        """Verify round-trip: parse → dump → expected top-level keys."""
        result = DocumentResult.model_validate_json(_RESULT_JSON)
        data = json.loads(result.model_dump_json())
        assert set(data.keys()) == {
            "document_id",
            "filename_doc_type",
            "classification",
            "confidence",
            "signals",
            "quality_assessment",
            "ocr_text",
            "processing_metadata",
        }

    def test_internal_scores_not_in_jsonl_output(self) -> None:
        """Internal quality fields must never appear when writing JSONL."""
        result = DocumentResult.model_validate_json(_RESULT_JSON)
        data = json.loads(result.model_dump_json())
        qa = data["quality_assessment"]
        assert "quality_score" not in qa
        assert "blur_score" not in qa
        assert "contrast_score" not in qa

    def test_round_trip(self) -> None:
        """Parse from JSON → dump → re-parse — document_id must survive."""
        original = DocumentResult.model_validate_json(_RESULT_JSON)
        restored = DocumentResult.model_validate_json(original.model_dump_json())
        assert original.document_id == restored.document_id
        assert original.classification.primary_class == restored.classification.primary_class
        assert original.confidence == restored.confidence

    def test_invalid_confidence_rejected(self) -> None:
        bad = json.loads(_RESULT_JSON)
        bad["confidence"] = 1.5
        with pytest.raises(ValidationError):
            DocumentResult.model_validate(bad)

    def test_default_processing_metadata(self) -> None:
        """processing_metadata should default to empty pages and 0ms."""
        data = json.loads(_RESULT_JSON)
        del data["processing_metadata"]
        result = DocumentResult.model_validate(data)
        assert result.processing_metadata.pages_used == []
        assert result.processing_metadata.processing_time_ms == 0


# ---------------------------------------------------------------------------
# BatchResult & DocumentError
# ---------------------------------------------------------------------------


class TestBatchResult:

    def test_parse_from_dict(self) -> None:
        data = {"total_documents": 0, "successful": 0, "failed": 0}
        batch = BatchResult.model_validate(data)
        assert batch.results == []
        assert batch.errors == []

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BatchResult.model_validate({"total_documents": -1, "successful": 0, "failed": 0})


class TestDocumentError:

    def test_parse_from_dict(self) -> None:
        data = {
            "document_id": "bad_doc",
            "error_type": "FileNotFoundError",
            "error_message": "File does not exist",
            "stage": "ocr",
        }
        err = DocumentError.model_validate(data)
        assert err.stage == "ocr"
        assert err.error_type == "FileNotFoundError"

    def test_parse_from_json_string(self) -> None:
        json_str = '{"document_id": "x", "error_type": "TimeoutError", "error_message": "timed out", "stage": "llm"}'
        err = DocumentError.model_validate_json(json_str)
        assert err.stage == "llm"

    def test_missing_stage_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DocumentError.model_validate(
                {"document_id": "x", "error_type": "E", "error_message": "msg"}
            )
