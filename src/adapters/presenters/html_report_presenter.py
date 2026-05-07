from __future__ import annotations

import base64
import io
import pathlib
from typing import Any

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

from src.domain.models.multi_page import MultiPageResult, PageClassificationResult

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    * { font-family: 'Outfit', sans-serif; box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: #0b0e14; color: #e1e1e1; }
    
    .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
    .header { text-align: center; margin-bottom: 50px; }
    .header h1 { font-size: 32px; font-weight: 600; color: #fff; margin-bottom: 10px; }
    .header p { color: #888; font-size: 16px; }

    .page-card { 
        background: #161b22; 
        border: 1px solid #30363d; 
        border-radius: 16px; 
        margin-bottom: 40px; 
        overflow: hidden;
        display: flex;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .viewer { flex: 1; background: #000; display: flex; align-items: center; justify-content: center; min-height: 600px; }
    .viewer img { max-width: 100%; height: auto; }

    .info { flex: 0.8; padding: 30px; border-left: 1px solid #30363d; }
    .page-num { font-size: 14px; color: #58a6ff; font-weight: 600; text-transform: uppercase; margin-bottom: 20px; }
    
    .badge { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-right: 8px; }
    .badge-root { background: #1f6feb; color: #fff; }
    .badge-sub { background: #238636; color: #fff; }
    .badge-uncertain { background: #d29922; color: #000; }

    .metric-row { display: flex; justify-content: space-between; margin: 15px 0; font-size: 14px; }
    .metric-label { color: #8b949e; }
    .metric-value { color: #c9d1d9; font-weight: 600; }

    .ocr-box { 
        background: #0d1117; 
        border: 1px solid #21262d; 
        border-radius: 8px; 
        padding: 15px; 
        font-size: 12px; 
        color: #8b949e; 
        height: 200px; 
        overflow-y: auto; 
        margin-top: 20px;
        white-space: pre-wrap;
    }
    
    .trail { font-size: 12px; color: #6e7681; margin-top: 10px; font-style: italic; }
</style>
"""

class HtmlReportPresenter:
    def generate_report(self, result: MultiPageResult, file_path: str) -> str:
        filename = result.file_name
        
        # 1. Convert file to images
        images = []
        try:
            p = pathlib.Path(file_path)
            if p.suffix.lower() == ".pdf" and PYMUPDF_AVAILABLE:
                doc = fitz.open(str(p))
                for page in doc:
                    pix = page.get_pixmap(dpi=120)
                    images.append(base64.b64encode(pix.tobytes("png")).decode())
            elif p.suffix.lower() in [".png", ".jpg", ".jpeg"] and PILLOW_AVAILABLE:
                with open(p, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode())
        except Exception:
            pass

        # 2. Render cards
        cards_html = ""
        for i, page in enumerate(result.pages):
            img_src = images[i] if i < len(images) else ""
            
            uncertain_tag = '<span class="badge badge-uncertain">UNCERTAIN</span>' if page.is_uncertain else ""
            
            cards_html += f"""
            <div class="page-card">
                <div class="viewer">
                    <img src="data:image/png;base64,{img_src}" />
                </div>
                <div class="info">
                    <div class="page-num">Page {page.page_index + 1}</div>
                    <div style="margin-bottom: 25px;">
                        <span class="badge badge-root">{page.root_code}</span>
                        <span class="badge badge-sub">{page.sub_code}</span>
                        {uncertain_tag}
                    </div>
                    
                    <div class="metric-row">
                        <span class="metric-label">Root Confidence</span>
                        <span class="metric-value">{page.root_confidence_pct:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Root Margin</span>
                        <span class="metric-value">{page.root_margin:.2f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Sub Confidence</span>
                        <span class="metric-value">{page.sub_confidence_pct:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Sub Margin</span>
                        <span class="metric-value">{page.sub_margin:.2f}</span>
                    </div>
                    <div class="ocr-box">{page.ocr_text}</div>
                    <div class="trail">Path: {" -> ".join(page.execution_trail)}</div>
                </div>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>DocGuru Report: {filename}</title>
            {_CSS}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Classification Report</h1>
                    <p>File: {filename} | Processing Time: {result.processing_time_ms}ms</p>
                </div>
                {cards_html}
            </div>
        </body>
        </html>
        """
        return html
