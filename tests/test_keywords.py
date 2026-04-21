"""Tests for keyword scorer."""

from classifier.keywords import (
    compute_keyword_scores,
    get_best_keyword_match,
    get_keyword_score_for_class,
)
from schemas.models import Subcategory


class TestComputeKeywordScores:

    def test_lab_keywords_english(self) -> None:
        text = "Laboratory Report - Blood Test Results: CBC, glucose, hemoglobin"
        scores = compute_keyword_scores(text)
        assert scores[Subcategory.LAB] > 0.5

    def test_lab_keywords_thai(self) -> None:
        text = "ผลตรวจเลือด จาก ห้องปฏิบัติการ"
        scores = compute_keyword_scores(text)
        assert scores[Subcategory.LAB] > 0.0

    def test_health_check_keywords(self) -> None:
        text = "Annual Health Check - Physical Examination - Health Screening Results"
        scores = compute_keyword_scores(text)
        assert scores[Subcategory.HEALTH_CHECK] > scores[Subcategory.LAB]

    def test_id_document_keywords(self) -> None:
        text = "National ID Card - Identification Number - Citizen Registration"
        scores = compute_keyword_scores(text)
        assert scores[Subcategory.ID] > 0.4

    def test_financial_keywords(self) -> None:
        text = "Invoice #12345 - Payment Receipt - Bank Statement - Tax ID"
        scores = compute_keyword_scores(text)
        assert scores[Subcategory.FINANCIAL] > 0.5

    def test_empty_text(self) -> None:
        scores = compute_keyword_scores("")
        assert all(score == 0.0 for score in scores.values())

    def test_irrelevant_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        scores = compute_keyword_scores(text)
        assert all(score <= 0.2 for score in scores.values())

    def test_score_capped_at_one(self) -> None:
        text = (
            "laboratory lab result lab report blood test CBC glucose "
            "cholesterol hemoglobin creatinine urinalysis"
        )
        scores = compute_keyword_scores(text)
        assert scores[Subcategory.LAB] <= 1.0

    def test_case_insensitive(self) -> None:
        scores_upper = compute_keyword_scores("LABORATORY BLOOD TEST CBC")
        scores_lower = compute_keyword_scores("laboratory blood test cbc")
        assert scores_upper[Subcategory.LAB] == scores_lower[Subcategory.LAB]

    def test_all_subcategories_present(self) -> None:
        scores = compute_keyword_scores("test")
        for subcategory in Subcategory:
            assert subcategory in scores


class TestGetBestKeywordMatch:

    def test_returns_best_subcategory(self) -> None:
        text = "Laboratory Blood Test CBC glucose hemoglobin creatinine"
        best_sub, best_score = get_best_keyword_match(text)
        assert best_sub == Subcategory.LAB
        assert best_score > 0.5

    def test_returns_tuple(self) -> None:
        sub, score = get_best_keyword_match("some text")
        assert isinstance(sub, Subcategory)
        assert isinstance(score, float)


class TestGetKeywordScoreForClass:

    def test_matching_class(self) -> None:
        text = "Laboratory Blood Test CBC glucose hemoglobin"
        score = get_keyword_score_for_class(text, Subcategory.LAB)
        assert score > 0.5

    def test_non_matching_class(self) -> None:
        text = "Laboratory Blood Test CBC glucose hemoglobin"
        score = get_keyword_score_for_class(text, Subcategory.ID)
        assert score == 0.0

    def test_score_range(self) -> None:
        text = "laboratory blood test CBC glucose hemoglobin creatinine urinalysis"
        for subcategory in Subcategory:
            score = get_keyword_score_for_class(text, subcategory)
            assert 0.0 <= score <= 1.0
