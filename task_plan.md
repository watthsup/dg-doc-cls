### Detailed Project Task Breakdown

| ID | Phase & Task Description | Sub-Tasks / Technical Requirements | Complexity | Est. MD |
| :--- | :--- | :--- | :---: | :---: |
| **0.0** | **Document Pre-processing** | | | **3.0** |
| 0.1 | Document Loader Adapter | Implement support for local/S3 file loading and Mime-type validation. | | 0.5 |
| 0.2 | Page Splitting Logic | Logic to split multi-page PDFs into individual images/sub-PDFs for processing. | | 0.5 |
| 0.3 | Parallel OCR Orchestration | Implementation of concurrent page-level OCR using Azure DI. | | 1.0 |
| 0.4 | OCR Result Aggregator | Merging page-level text, confidence, and metadata into a unified document object. | | 0.5 |
| 0.5 | Pre-processing Unit Tests | Validation of splitting, merging, and handling of corrupted/blurred files. | | 0.5 |
| **1.0** | **Architecture & Foundation** | | | **3.5** |
| 1.1 | Graph State Design | Define `GraphState`, Pydantic schemas for node outputs. | | 0.5 |
| 1.2 | Logprob Analyzer Engine | Implement math logic for Margin calculation and confidence normalization. | | 1.0 |
| 1.3 | OCR Integration Layer | Implement Azure DI wrapper with `ainvoke` support and error handling. | | 0.5 |
| 1.4 | Infrastructure Utilities | Setup Structlog, Pydantic-settings, and LangSmith tracing. | | 1.0 |
| 1.5 | Base Unit Tests | Setup pytest for core utility functions (Logprob, OCR). | | 0.5 |
| **2.0** | **Taxonomy & Specialist Nodes** | | **High** | **6.0** |
| 2.1 | Root Router Implementation | Prompt engineering for MED vs NON classification. | | 1.0 |
| 2.2 | Medical Specialist (Sub-codes) | Implement medical node; Prompt tuning for LAB, CHK, OTH. | **[ * ]** | 2.0 |
| 2.3 | Non-Med Specialist (Sub-codes) | Implement non-med node; Prompt tuning for ID, PAS, OTH. | **[ * ]** | 2.0 |
| 2.4 | Pydantic Output Validation | Implement strict schema validation for LLM classification codes. | | 0.5 |
| 2.5 | Node Logic Unit Tests | Verify state updates, confidence calculation, and fallback behavior for all categories. | | 0.5 |
| **3.0** | **Active Labeling Strategy** | | | **5.5** |
| 3.1 | Pilot Batch Processing | Script to run 1,000+ docs through current pipeline; export data. | | 1.0 |
| 3.2 | Verification Tooling | Build Streamlit UI for fast human verification (Image vs Prediction). | | 1.5 |
| 3.3 | Human Labeling Execution | (External) Users verify predictions to create Gold Standard labels. | **[ * ]** | 2.0 |
| 3.4 | Ground Truth Sanitization | Script to resolve label conflicts and clean final evaluation set. | | 1.0 |
| **4.0** | **Reliability Engineering** | | **High** | **6.0** |
| 4.1 | Confusion Matrix Analysis | Evaluate against Gold Standard; find high False Positive classes. | | 1.0 |
| 4.2 | Margin Threshold Optimization | Plot precision/recall curves to find the optimal Margin Threshold. | | 2.0 |
| 4.3 | Prompt Conflict Resolution | Re-tune prompts to fix overlaps found during evaluation. | **[ * ]** | 2.0 |
| 4.4 | Failure Case Documentation | Create "Negative Examples" dataset for targeted performance tuning. | | 1.0 |
| **5.0** | **API Layer & Wrapper** | | | **2.0** |
| 5.1 | FastAPI Endpoints | `/classify` (async), `/health`, and metadata endpoints. | | 1.0 |
| 5.2 | Validation Middleware | Input validation (File size, Mime type) and Global Exception Handling. | | 0.5 |
| 5.3 | Integration Docs | Generate Swagger/Redoc and provide Client-side SDK samples. | | 0.5 |
| **TOTAL** | | | | **26.0 MD** |

---

### Project Remarks & Constraints

#### 1. Taxonomy Complexity Scaling `[ * ]`
Tasks marked with `[ * ]` are directly impacted by the number of classes. 
*   **Current assumption:** ~10-15 classes total.
*   **Risk:** If the taxonomy grows to 30-50 classes, the time for **Human Labeling (3.3)** and **Prompt Conflict Resolution (4.3)** will increase exponentially because the "boundary" between classes becomes harder to define.

#### 2. Ground Truth as the Primary Bottleneck
The absence of "Gold Standard" labels is the most critical risk to system reliability. Phase 3.0 (Labeling) is a mandatory prerequisite for Phase 4.0; without verified answers, we cannot scientifically optimize the system for maximum accuracy or detect subtle model errors.

#### 3. Labeling Workflow Efficiency
To hit the estimated timeline in Phase 3.3, the verification process in Phase 3.2 must be optimized for speed. A high-throughput labeling environment is required to build a statistically significant dataset within the allocated 2.0 MD.