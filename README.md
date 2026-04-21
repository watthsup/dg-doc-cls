# Document Classification & Metadata Extraction System

Classifies document files (PDF/images) into predefined categories, extracts hospital names via LLM, and provides reliable, explainable confidence scores.

## Features

- **2-level classification**: Medical/Non-Medical → Lab/Health Check/Medical Other/ID/Financial/Other
- **Hospital name extraction** via LLM structured output (single call)
- **Explainable confidence scoring** from observable signals (no LLM confidence)
- **Batch processing** with concurrent document processing
- **Azure Document Intelligence** OCR with word-level confidence
- **Azure OpenAI Foundry** for classification via Pydantic structured output
- **Export** to JSONL (primary) and CSV (optional)

## Architecture

```
config/         → settings + logging
schemas/        → Pydantic models (input/output contracts)
ocr/            → Azure Document Intelligence + quality assessment
classifier/     → Azure OpenAI LLM + keyword scoring + prompts
confidence/     → Composite confidence calculation
pipeline/       → Per-document + batch orchestration
exporters/      → JSONL/CSV output
```

## Quick Start

### Prerequisites

- Python 3.12+
- Azure Document Intelligence resource
- Azure OpenAI resource with a deployed model

### Setup

```bash
# Activate conda environment
conda activate doc_cls

# Install the package
pip install -e ".[dev]"

# Create .env from template
cp .env.example .env
# Edit .env and add your Azure credentials
```

### Run

```bash
# Classify all documents in a folder
python scripts/run_batch.py --input-dir ./documents --output-dir ./output

# With options
python scripts/run_batch.py \
  --input-dir ./documents \
  --output-dir ./output \
  --output-format both \
  --max-concurrency 10 \
  --max-pages 3 \
  --verbose
```

### Run Tests

```bash
# Unit tests (no Azure credentials needed)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=.
```

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_DI_ENDPOINT` | (required) | Document Intelligence endpoint URL |
| `AZURE_DI_KEY` | (required) | Document Intelligence API key |
| `AZURE_DI_MODEL` | `prebuilt-layout` | Azure DI model ID |
| `AZURE_OPENAI_ENDPOINT` | (required) | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | (required) | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o-mini` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-05-01-preview` | API version |
| `MAX_PAGES` | `2` | Max pages to OCR per document |
| `MAX_CONCURRENCY` | `5` | Max parallel documents |
| `LOG_FORMAT` | `console` | `console` or `json` |
| `OUTPUT_FORMAT` | `jsonl` | `jsonl`, `csv`, or `both` |

## Output Schema

Each document produces a `DocumentResult`:

```json
{
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
    "skew_angle": 0.0
  },
  "processing_metadata": {
    "pages_used": [0, 1],
    "processing_time_ms": 3500
  }
}
```

## Project Structure

```
docguru-doc-cls/
├── config/
│   ├── settings.py          # Pydantic Settings (Azure config)
│   └── logging.py           # structlog configuration
├── schemas/
│   └── models.py            # All Pydantic models + LLMOutput
├── ocr/
│   ├── engine.py            # Azure Document Intelligence client
│   ├── page_sampler.py      # Page selection
│   └── quality.py           # OCR-confidence-based quality
├── classifier/
│   ├── llm.py               # AzureChatOpenAI + structured output
│   ├── prompts.py           # Classification prompt template
│   └── keywords.py          # Keyword-lite scoring
├── confidence/
│   └── calculator.py        # Weighted composite confidence
├── pipeline/
│   ├── document.py          # Per-document pipeline
│   ├── batch.py             # Concurrent batch runner
│   └── filesystem.py        # Directory scanner
├── exporters/
│   └── writer.py            # JSONL/CSV export
├── scripts/
│   └── run_batch.py         # CLI entry point
└── tests/                   # 49 unit tests
```
