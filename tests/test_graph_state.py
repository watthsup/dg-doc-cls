"""Tests for graph/state.py and schemas/models.py graph-related additions."""

from __future__ import annotations

import pytest

from graph.state import GraphState, create_initial_state
from schemas.models import (
    CODE_TO_PRIMARY,
    CODE_TO_SUBCATEGORY,
    PRIMARY_TO_CODE,
    SUBCATEGORY_TO_CODE,
    VALID_MED_SUB_CODES,
    VALID_NONMED_SUB_CODES,
    VALID_ROOT_CODES,
    ClassificationCode,
    LogprobAnalysis,
    PrimaryClass,
    Subcategory,
)


class TestClassificationCode:
    """Test the 3-letter code enum."""

    def test_all_codes_exist(self) -> None:
        """All 12 codes should be present."""
        assert len(ClassificationCode) == 12

    def test_root_codes(self) -> None:
        assert ClassificationCode.MED == "MED"
        assert ClassificationCode.NON == "NON"

    def test_medical_sub_codes(self) -> None:
        med_codes = {"LAB", "HCK", "IMG", "IPD", "MCE", "DIS", "MOT"}
        actual = {c.value for c in VALID_MED_SUB_CODES}
        assert actual == med_codes

    def test_nonmed_sub_codes(self) -> None:
        nonmed_codes = {"ID_", "FIN", "OTH"}
        actual = {c.value for c in VALID_NONMED_SUB_CODES}
        assert actual == nonmed_codes

    def test_root_codes_set(self) -> None:
        assert VALID_ROOT_CODES == {ClassificationCode.MED, ClassificationCode.NON}

    def test_no_overlap_between_groups(self) -> None:
        """Root, med-sub, and nonmed-sub must not overlap."""
        root_vals = {c.value for c in VALID_ROOT_CODES}
        med_vals = {c.value for c in VALID_MED_SUB_CODES}
        nonmed_vals = {c.value for c in VALID_NONMED_SUB_CODES}

        assert root_vals.isdisjoint(med_vals)
        assert root_vals.isdisjoint(nonmed_vals)
        assert med_vals.isdisjoint(nonmed_vals)

    def test_all_codes_covered(self) -> None:
        """Every ClassificationCode should be in exactly one group."""
        all_grouped = VALID_ROOT_CODES | VALID_MED_SUB_CODES | VALID_NONMED_SUB_CODES
        assert all_grouped == set(ClassificationCode)


class TestCodeMappings:
    """Test bidirectional code <-> enum mappings."""

    def test_code_to_primary_covers_root_codes(self) -> None:
        assert set(CODE_TO_PRIMARY.keys()) == VALID_ROOT_CODES

    def test_code_to_subcategory_covers_sub_codes(self) -> None:
        expected = VALID_MED_SUB_CODES | VALID_NONMED_SUB_CODES
        assert set(CODE_TO_SUBCATEGORY.keys()) == expected

    def test_primary_to_code_inverse(self) -> None:
        """PRIMARY_TO_CODE should be the inverse of CODE_TO_PRIMARY."""
        for code, primary in CODE_TO_PRIMARY.items():
            assert PRIMARY_TO_CODE[primary] == code

    def test_subcategory_to_code_inverse(self) -> None:
        """SUBCATEGORY_TO_CODE should be the inverse of CODE_TO_SUBCATEGORY."""
        for code, sub in CODE_TO_SUBCATEGORY.items():
            assert SUBCATEGORY_TO_CODE[sub] == code

    def test_lab_maps_correctly(self) -> None:
        assert CODE_TO_SUBCATEGORY[ClassificationCode.LAB] == Subcategory.LAB

    def test_hck_maps_correctly(self) -> None:
        assert CODE_TO_SUBCATEGORY[ClassificationCode.HCK] == Subcategory.HEALTH_CHECK

    def test_fin_maps_correctly(self) -> None:
        assert CODE_TO_SUBCATEGORY[ClassificationCode.FIN] == Subcategory.FINANCIAL

    def test_id_maps_correctly(self) -> None:
        assert CODE_TO_SUBCATEGORY[ClassificationCode.ID_] == Subcategory.ID

    def test_med_maps_to_medical(self) -> None:
        assert CODE_TO_PRIMARY[ClassificationCode.MED] == PrimaryClass.MEDICAL

    def test_non_maps_to_non_medical(self) -> None:
        assert CODE_TO_PRIMARY[ClassificationCode.NON] == PrimaryClass.NON_MEDICAL


class TestLogprobAnalysis:
    """Test the LogprobAnalysis model."""

    def test_parse_from_dict(self) -> None:
        data = {
            "top1_token": "MED",
            "top1_logprob": -0.01,
            "top2_token": "NON",
            "top2_logprob": -4.5,
            "margin_score": 4.49,
            "confidence_pct": 99.0,
        }
        analysis = LogprobAnalysis.model_validate(data)
        assert analysis.top1_token == "MED"
        assert analysis.margin_score == 4.49

    def test_confidence_pct_bounds(self) -> None:
        with pytest.raises(Exception):
            LogprobAnalysis(
                top1_token="MED",
                top1_logprob=-0.01,
                top2_token="NON",
                top2_logprob=-4.5,
                margin_score=4.49,
                confidence_pct=150.0,  # Over 100
            )


class TestGraphState:
    """Test the GraphState factory function."""

    def test_create_initial_state(self) -> None:
        state = create_initial_state(
            document_id="test_001",
            file_path="/tmp/test.pdf",
            file_type="pdf",
        )
        assert state["document_id"] == "test_001"
        assert state["file_path"] == "/tmp/test.pdf"
        assert state["file_type"] == "pdf"
        assert state["root_code"] is None
        assert state["sub_code"] is None
        assert state["is_uncertain"] is False
        assert state["requires_human_review"] is False
        assert state["execution_trail"] == []

    def test_initial_state_margins_are_zero(self) -> None:
        state = create_initial_state("id", "/path", "pdf")
        assert state["root_margin"] == 0.0
        assert state["sub_margin"] == 0.0
        assert state["root_confidence_pct"] == 0.0
        assert state["sub_confidence_pct"] == 0.0
