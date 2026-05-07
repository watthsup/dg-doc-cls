from __future__ import annotations
import time
import asyncio
from pathlib import Path
from src.adapters.outbound.llm.client import LLMClassifier
from src.adapters.outbound.ocr.client import analyze_document, create_di_client
from src.domain.models.multi_page import MultiPageResult, PageClassificationResult
from src.infrastructure.config.settings import AppConfig

class VanillaRunner:
    """Legacy runner: Direct single-hop LLM classification."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.classifier = LLMClassifier(config)
        self.di_client = create_di_client(config)

    async def run(self, file_path: str, document_id: str = "vanilla_doc") -> MultiPageResult:
        start_time = time.perf_counter()
        
        # 1. OCR (whole document)
        ocr_result = await asyncio.to_thread(
            analyze_document,
            client=self.di_client,
            file_path=Path(file_path),
            model_id=self.config.azure_di_model,
        )
        
        # 2. Sequential Classification
        pages = []
        for ocr_page in ocr_result.pages:
            llm_output = await self.classifier.classify(ocr_page.text)
            
            # Map LLMOutput to PageClassificationResult
            pages.append(PageClassificationResult(
                page_index=ocr_page.page_index,
                root_code=llm_output.primary_class.value.upper(),
                sub_code=llm_output.subcategory.value.upper(),
                hospital_name=llm_output.hospital_name,
                ocr_text=ocr_page.text,
                execution_trail=["vanilla_direct_call"],
                is_uncertain=False,  # Vanilla doesn't have margin scoring
                root_score=1.0,
                sub_score=1.0,
                root_confidence_pct=100.0,
                sub_confidence_pct=100.0
            ))
            
        processing_time = int((time.perf_counter() - start_time) * 1000)
        
        return MultiPageResult(
            document_id=document_id,
            file_name=Path(file_path).name,
            total_pages=len(pages),
            pages=pages,
            processing_time_ms=processing_time,
        )

def build_vanilla_runner(config: AppConfig) -> VanillaRunner:
    return VanillaRunner(config)
