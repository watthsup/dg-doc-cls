# DocGuru — Document Classification & Metadata Extraction

An enterprise-grade document sorting gateway that classifies high-volume incoming documents via a **Hierarchical LangGraph Pipeline** with mandatory statistical confidence analysis and Human-in-the-Loop (HITL) verification.

---

## Features

- **Hierarchical 2-stage classification** — Root Router (MED/NON) → Specialist Sub-Classifier
- **Logprob-based Reliability Engine** — Margin = `ln(P_top1) − ln(P_top2)`; auto-flags low-confidence docs
- **Human-in-the-Loop (HITL)** — LangGraph interrupt when `margin < 1.5`; resume from SQLite checkpoint
- **Hospital name extraction** — Concurrent async LLM call per medical document
- **Explainable confidence scoring** — From observable signals only (OCR + image quality)
- **Azure Document Intelligence OCR** — Word-level confidence, multi-page support
- **Batch processing** — Concurrent with semaphore control; isolated per-task SDK clients
- **Export** — JSONL (primary) and CSV

---

## Classification Taxonomy

### V2 — LangGraph Hierarchical (recommended)

```
Root
├── MED (Medical)
│   ├── LAB  — Lab test results (CBC, LFT, Lipid, etc.)
│   ├── CHK  — Health check / wellness / executive screening
│   ├── CLI  — Clinical document (OPD/IPD notes, discharge summary, etc.)
│   └── OTH  — Medical Other (imaging, certificates, etc.) ← fallback
│
└── NON (Non-Medical)
    ├── PS  — Passport (international travel document, MRZ)
    ├── ID  — National ID card / driver's licence
    ├── FIN  — Financial (invoice, receipt, bank statement, hospital bill)
    └── OTH  — Other ← fallback
```

> **Scope note:** Medical sub-classification focuses on the two highest-volume categories
> (LAB & CHK). All other medical documents fall to OTH for downstream specialist handling.
> Passport is explicitly separated from National ID because they have distinct key fields
> and different downstream business rules.

### V1 — Direct LLM (legacy, still supported)

| Primary | Subcategory |
|---------|-------------|
| medical | lab, health_check, imaging_report, ipd_opd_document, medical_certificate, discharge_summary, medical_other |
| non_medical | id, financial, other |

---

## Architecture

```
src/infrastructure/  → Frameworks & drivers (Config, Telemetry)
src/adapters/        → Interface adapters
  inbound/             → CLI, API, UI routes (Streamlit)
  outbound/            → LLM and OCR clients
  orchestration/       → LangGraph nodes, edges, state
  presenters/          → CSV exporters
src/application/     → Use cases & ports
  ports/               → Input/Output Protocol interfaces
  use_cases/           → ClassifyPageService, Pipeline workflows
  contracts/           → LLM payload schemas
  prompts/             → LLM prompt templates
src/domain/          → Entities & pure policies
  models/              → Domain data classes and enums
  services/            → Logprob analyzer, Confidence calculator
src/shared/          → Pure utilities
src/bootstrap.py     → Composition root
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Azure Document Intelligence resource
- Azure OpenAI resource (deployed model with `logprobs` support)
- `conda` environment: `doc_cls`

### Setup

```bash
uv venv
uv sync --all-extras
cp .env.example .env
# Edit .env with your Azure credentials
```

---

## Running

### Unified CLI (Recommended)

After installation, the `docguru` command is available. Alternatively, you can run it via `python -m src.main`.

#### Process Documents (Classify)
Classify one or more documents and export results to CSV/JSONL.

```bash
# Explicit call (recommended)
uv run python -m src.main process --file path/to/doc.pdf --type doc_cls

# Vanilla mode (Direct LLM)
uv run python -m src.main process --file path/to/doc.pdf --type vanilla

# Using entry point
uv run docguru process --dir ./documents --out ./results --concurrency 5

# Debug mode
uv run docguru process --file doc.pdf --verbose
```

#### Generate Interactive Reports
Run classification and generate self-contained HTML diagnostic reports for visual review.

```bash
uv run python -m src.main gen-report --dir ./documents --out ./html_reports --type doc_cls
```

### Streamlit UI
```bash
uv run streamlit run app.py
```
- **Main page** — Upload a document; toggle V1/V2 pipeline in the sidebar
- **Review page** — Resolve HITL-flagged documents from the checkpoint database

---

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_DI_ENDPOINT` | *(required)* | Document Intelligence endpoint URL |
| `AZURE_DI_MODEL` | `prebuilt-layout` | Azure DI model ID |
| `AZURE_OPENAI_ENDPOINT` | *(required)* | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o-mini` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-05-01-preview` | API version |
| `MAX_PAGES` | `2` | Max pages to OCR per document |
| `MAX_CONCURRENCY` | `5` | Max parallel documents |
| `LOG_FORMAT` | `console` | `console` or `json` |
| `OUTPUT_FORMAT` | `jsonl` | `jsonl`, `csv`, or `both` |
| `MARGIN_THRESHOLD` | `1.5` | Logprob margin below this triggers HITL |
| `CHECKPOINT_DB_PATH` | `./checkpoints/doc_cls.db` | SQLite checkpoint path (V2 only) |
| `LOGPROBS_TOP_N` | `3` | Number of top logprobs to capture |

Authentication uses **Azure Managed Identity** (`DefaultAzureCredential`). No API keys in environment.

---

## Output Schema

Each document produces a `DocumentResult`:

```json
{
  "document_id": "report_abc123ef",
  "file_name": "lab_result.pdf",
  "hospital_name": "Bumrungrad International Hospital",
  "pages": [
    {
      "page_index": 0,
      "classification": {
        "primary_class": "medical",
        "subcategory": "lab",
        "hospital_name": "Bumrungrad International Hospital"
      },
      "confidence": 0.87,
      "signals": {
        "ocr_confidence": 0.91,
        "quality_score": 0.82
      },
      "quality_assessment": {
        "issues": [],
        "skew_angle": 0.3
      }
    }
  ],
  "processing_metadata": {
    "pages_used": [0],
    "total_pages": 2,
    "processing_time_ms": 3400
  }
}
```

---

## Testing

```bash
# All tests (no Azure credentials required)
uv run pytest tests/ -v

# Graph-specific tests only
uv run pytest tests/test_logprob_analyzer.py tests/test_graph_state.py -v

# With coverage
uv run pytest tests/ -v --cov=.
```

Current: **131 tests, all passing**.

---

## Project Structure

```
docguru-doc-cls/
├── src/
│   ├── domain/               Entities & pure policies
│   │   ├── models/           DocumentType, LogprobAnalysis, etc.
│   │   └── services/         Logprob analyzer, confidence calculator
│   ├── application/          Use cases & ports
│   │   ├── contracts/        LLM output schemas
│   │   ├── ports/            Input/Output interfaces
│   │   ├── prompts/          Classification prompt templates
│   │   └── use_cases/        ClassifyPageService, pipelines
│   ├── adapters/             Interface adapters
│   │   ├── inbound/          CLI and UI
│   │   ├── outbound/         LLM and OCR clients
│   │   ├── orchestration/    LangGraph nodes, edges, state
│   │   └── presenters/       CSV and JSONL exporters
│   ├── infrastructure/       Frameworks & drivers
│   │   ├── config/           Pydantic settings
│   │   └── telemetry/        structlog setup
│   └── bootstrap.py          Composition root
├── tests/                    Unit tests
├── scripts/                  CLI entry points
├── pages/                    Streamlit HITL review page
└── app.py                    Streamlit main app
```
