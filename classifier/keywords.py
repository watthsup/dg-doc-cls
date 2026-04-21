"""Keyword-lite scoring — lightweight classification signal.

Simple substring matching against per-class keyword lists.
Deliberately NOT NER or NLP — fast, low-maintenance.
"""

from __future__ import annotations

from schemas.models import Subcategory

KEYWORD_LISTS: dict[Subcategory, list[str]] = {
    Subcategory.LAB: [
        "laboratory", "lab result", "lab report", "blood test",
        "CBC", "glucose", "cholesterol", "hemoglobin", "creatinine",
        "urinalysis", "ผลตรวจเลือด", "ผลแลป", "ห้องปฏิบัติการ",
        "ผลตรวจทางห้องปฏิบัติการ",
    ],
    Subcategory.HEALTH_CHECK: [
        "health check", "health examination", "annual checkup",
        "physical examination", "medical checkup", "health screening",
        "ตรวจสุขภาพ", "ผลตรวจสุขภาพ", "ตรวจสุขภาพประจำปี",
        "ตรวจร่างกาย", "สุขภาพทั่วไป",
    ],
    Subcategory.MEDICAL_OTHER: [
        "medical report", "medical certificate", "diagnosis",
        "treatment", "prescription", "discharge summary",
        "clinical", "patient", "ใบรับรองแพทย์", "วินิจฉัย",
        "การรักษา", "ใบสั่งยา", "สรุปการรักษา",
    ],
    Subcategory.ID: [
        "identification", "ID card", "passport", "driver license",
        "national ID", "citizen", "บัตรประชาชน", "หนังสือเดินทาง",
        "ใบขับขี่", "เลขบัตรประชาชน", "เลขประจำตัว",
    ],
    Subcategory.FINANCIAL: [
        "invoice", "receipt", "bank statement", "financial",
        "payment", "tax", "account", "balance", "ใบเสร็จ",
        "ใบแจ้งหนี้", "ใบกำกับภาษี", "การเงิน", "ธนาคาร",
    ],
    Subcategory.OTHER: [
        "contract", "agreement", "letter", "memo", "notice",
        "certificate", "สัญญา", "หนังสือ", "ประกาศ", "บันทึก",
    ],
}

_MATCH_DENOMINATOR: int = 5


def _count_keyword_matches(text: str, keywords: list[str]) -> int:
    """Count keyword presence (not frequency) — case-insensitive."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def compute_keyword_scores(text: str) -> dict[Subcategory, float]:
    """Compute keyword match score for each subcategory.

    Formula: min(1.0, matched_keywords / 5)
    """
    scores: dict[Subcategory, float] = {}
    for subcategory, keywords in KEYWORD_LISTS.items():
        matches = _count_keyword_matches(text, keywords)
        scores[subcategory] = min(1.0, matches / _MATCH_DENOMINATOR)
    return scores


def get_best_keyword_match(text: str) -> tuple[Subcategory, float]:
    """Return the subcategory with highest keyword score."""
    scores = compute_keyword_scores(text)
    best_subcategory = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_subcategory, scores[best_subcategory]


def get_keyword_score_for_class(text: str, subcategory: Subcategory) -> float:
    """Get keyword score for a specific classified subcategory."""
    keywords = KEYWORD_LISTS.get(subcategory, [])
    matches = _count_keyword_matches(text, keywords)
    return min(1.0, matches / _MATCH_DENOMINATOR)
