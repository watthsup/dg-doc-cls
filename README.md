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
│   ├── HCK  — Health check / wellness / executive screening
│   └── MOT  — Medical Other (imaging, OPD notes, certs, etc.) ← fallback
│
└── NON (Non-Medical)
    ├── PAS  — Passport (international travel document, MRZ)
    ├── ID_  — National ID card / driver's licence
    ├── FIN  — Financial (invoice, receipt, bank statement, hospital bill)
    └── OTH  — Other ← fallback
```

> **Scope note:** Medical sub-classification focuses on the two highest-volume categories
> (LAB & HCK). All other medical documents fall to MOT for downstream specialist handling.
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
config/         → Pydantic Settings (Azure config, thresholds)
schemas/        → All Pydantic models + classification codes + enums
ocr/            → Azure Document Intelligence + quality assessment
classifier/     → V1 LLM classifier (direct structured output)
confidence/     → Composite confidence scoring
graph/          → V2 LangGraph pipeline
  ├── state.py          GraphState TypedDict
  ├── nodes.py          OCR ingestion, Root Router, Specialists, HITL gateway
  ├── logprob_analyzer  Reliability Engine (margin calculation)
  ├── prompts.py        Per-node prompt templates
  └── builder.py        Graph construction + SQLite checkpointer
pipeline/       → Per-document + batch orchestration (V1 & V2)
exporters/      → JSONL/CSV output
pages/          → Streamlit review page (HITL adjudication)
scripts/        → CLI entry points
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
conda activate doc_cls
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your Azure credentials
```

---

## Running

### V2 — LangGraph Pipeline (recommended)

#### Single document
```bash
python scripts/run_graph.py path/to/document.pdf
python scripts/run_graph.py path/to/document.pdf --verbose
python scripts/run_graph.py path/to/document.pdf --json
```

#### Batch (directory)
```bash
python scripts/run_graph_batch.py ./documents/
python scripts/run_graph_batch.py ./documents/ --output results_v2.jsonl --max-concurrency 3
```

### V1 — Direct LLM (legacy)

#### Single document
```bash
python scripts/run_single.py path/to/document.pdf
python scripts/run_single.py path/to/document.pdf --json
```

#### Batch
```bash
python scripts/run_batch.py --input-dir ./documents --output-dir ./results
python scripts/run_batch.py --input-dir ./documents --output-dir ./output \
  --output-format both --max-concurrency 10 --max-pages 3 --verbose
```

### Streamlit UI
```bash
streamlit run app.py
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
pytest tests/ -v

# Graph-specific tests only
pytest tests/test_logprob_analyzer.py tests/test_graph_state.py tests/test_graph_adapter.py -v

# With coverage
pytest tests/ -v --cov=.
```

Current: **131 tests, all passing**.

---

## Project Structure

```
docguru-doc-cls/
├── config/
│   ├── settings.py          Pydantic Settings (Azure + HITL config)
│   └── logging.py           structlog configuration
├── schemas/
│   └── models.py            All Pydantic models, ClassificationCode enum, mappings
├── ocr/
│   ├── engine.py            Azure Document Intelligence client
│   ├── page_sampler.py      Page selection strategy
│   └── quality.py           Image quality assessment (OpenCV)
├── classifier/              V1 direct LLM classifier
│   ├── llm.py
│   └── prompts.py
├── confidence/
│   └── calculator.py        Weighted composite confidence
├── graph/                   V2 LangGraph pipeline
│   ├── __init__.py
│   ├── state.py             GraphState TypedDict + factory
│   ├── logprob_analyzer.py  Reliability Engine (margin score)
│   ├── prompts.py           Per-node prompt templates
│   ├── nodes.py             Node functions (OCR, Router, Specialists, HITL)
│   └── builder.py           Graph construction + SQLite checkpointer
├── pipeline/
│   ├── document.py          V1 per-document pipeline
│   ├── batch.py             V1 concurrent batch runner
│   ├── graph_adapter.py     GraphState → PageResult/DocumentResult adapter
│   └── filesystem.py        Directory scanner + file-type detection
├── exporters/
│   └── writer.py            JSONL/CSV export
├── pages/
│   └── review.py            Streamlit HITL review page
├── scripts/
│   ├── run_single.py        V1 single-doc CLI
│   ├── run_batch.py         V1 batch CLI
│   ├── run_graph.py         V2 single-doc CLI (LangGraph)
│   └── run_graph_batch.py   V2 batch CLI (LangGraph)
├── app.py                   Streamlit main app (V1/V2 toggle)
└── tests/                   131 unit tests
```
