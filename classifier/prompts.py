"""Classification prompt template — externalized per agent_rules.md.

Single LLM call returns classification + hospital name extraction.
Temperature = 0 for deterministic output. No confidence requested.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a document classification expert. Your task is to classify a document \
based on its OCR-extracted text and extract relevant metadata.

## Classification Schema

### Level 1: Primary Class
- **medical**: Documents related to healthcare, medical tests, hospital records
- **non_medical**: All other documents (IDs, financial, general)

### Level 2: Subcategory

If primary_class is "medical":
- **lab**: Laboratory test results (blood tests, urinalysis, etc.)
- **health_check**: Annual health checkups, physical examinations
- **medical_other**: Medical certificates, prescriptions, \
discharge summaries, other medical docs

If primary_class is "non_medical":
- **id**: Identification documents (ID cards, passports, driver licenses)
- **financial**: Invoices, receipts, bank statements, tax documents
- **other**: Any document that doesn't fit the above categories

## Hospital Name Extraction
- If this is a medical document, extract the hospital or clinic name
- Look for names in the header, letterhead, or logo area
- Return null if not a medical document or if no hospital name is found

## Rules
1. Choose EXACTLY ONE primary_class and EXACTLY ONE subcategory
2. The subcategory MUST be valid for the chosen primary_class
3. If ambiguous, classify based on the DOMINANT content
4. A medical invoice = "financial" (non_medical) — classify by PURPOSE
5. If text is mostly unreadable or empty, classify as "other" (non_medical)
6. Do NOT explain your reasoning
7. Do NOT provide confidence scores
"""

CLASSIFICATION_USER_PROMPT = """\
Classify the following document and extract the hospital name if applicable:

---
{ocr_text}
---
"""

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFICATION_SYSTEM_PROMPT),
        ("human", CLASSIFICATION_USER_PROMPT),
    ]
)
