import asyncio
import tempfile
from pathlib import Path

import streamlit as st
import structlog

from config import AppConfig, setup_logging
from ocr.engine import analyze_document, create_di_client, load_document_images
from pipeline.document import DocumentProcessor
from pipeline.filesystem import detect_file_type, generate_document_id

st.set_page_config(page_title="DocGuru Classifier", layout="wide")

st.title("📄 DocGuru Document Classifier")

st.sidebar.header("Configuration")
max_pages = st.sidebar.number_input("Max Pages to Process", min_value=1, max_value=50, value=10)
pipeline_mode = st.sidebar.radio(
    "Pipeline Mode",
    ["V2 — LangGraph (Hierarchical)", "V1 — Direct LLM (Single-hop)"],
    index=0,
)

uploaded_file = st.file_uploader(
    "Upload Medical Document", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    status_msg = st.empty()
    status_msg.info(f"Preparing {uploaded_file.name}...")

    config = AppConfig()  # type: ignore[call-arg]
    config = config.model_copy(update={"max_pages": max_pages})
    setup_logging(config)
    log = structlog.get_logger()

    file_type = detect_file_type(tmp_path)
    if not file_type:
        st.error("Unsupported file type!")
        st.stop()

    # ===================================================================
    # V2 — LangGraph (Hierarchical, per-page)
    # ===================================================================
    if pipeline_mode.startswith("V2"):
        with st.spinner("🧠 LangGraph hierarchical classification..."):
            try:
                processor = DocumentProcessor(config=config)
                result = asyncio.run(processor.process_file(tmp_path))
                display_images = load_document_images(tmp_path, file_type)
            except Exception as e:
                log.exception("streamlit_v2_failed", error=str(e))
                st.error(f"Error processing document: {e}")
                st.stop()

        if result.has_uncertain_pages:
            status_msg.warning("⚠️ Some pages have low confidence — review recommended!")
        else:
            status_msg.success(f"✅ Classification Complete! ({result.processing_time_ms}ms)")

        st.markdown(f"**Summary:** `{result.summary}`")
        st.caption(f"Document ID: {result.document_id} | Total Pages: {result.total_pages}")
        st.markdown("---")

        for page in result.pages:
            st.subheader(f"Page {page.page_index + 1}")
            col_img, col_data = st.columns([1, 1])

            with col_img:
                if page.page_index < len(display_images):
                    st.image(display_images[page.page_index], use_container_width=True)

            with col_data:
                st.markdown("### 🏷️ Classification")
                st.success(f"**{page.root_code}** ➔ **{page.sub_code}**")

                if page.hospital_name:
                    st.info(f"🏥 **Hospital:** {page.hospital_name}")

                # Logprob Analysis
                st.markdown("### 🔬 Logprob Analysis")
                col_root, col_sub = st.columns(2)

                with col_root:
                    root_color = "green" if page.root_margin >= config.margin_threshold else "red"
                    st.markdown(f"**Root:** `{page.root_code}`")
                    st.markdown(f"Margin: :{root_color}[{page.root_margin:.3f}]")
                    st.markdown(f"Confidence: {page.root_confidence_pct:.1f}%")
                    if page.root_logprobs:
                        lp = page.root_logprobs
                        st.caption(
                            f"Top1: {lp.get('top1_token')} ({lp.get('top1_logprob', 0):.3f}) | "
                            f"Top2: {lp.get('top2_token')} ({lp.get('top2_logprob', 0):.3f})"
                        )

                with col_sub:
                    if page.sub_code:
                        sub_color = "green" if page.sub_margin >= config.margin_threshold else "red"
                        st.markdown(f"**Sub:** `{page.sub_code}`")
                        st.markdown(f"Margin: :{sub_color}[{page.sub_margin:.3f}]")
                        st.markdown(f"Confidence: {page.sub_confidence_pct:.1f}%")
                        if page.sub_logprobs:
                            lp = page.sub_logprobs
                            st.caption(
                                f"Top1: {lp.get('top1_token')} ({lp.get('top1_logprob', 0):.3f}) | "
                                f"Top2: {lp.get('top2_token')} ({lp.get('top2_logprob', 0):.3f})"
                            )
                    else:
                        st.info("Sub-classification not reached")

                if page.is_uncertain:
                    st.warning("⚠️ Flagged as uncertain — human review recommended.")

                st.caption(f"Trail: {' → '.join(page.execution_trail)}")

                with st.expander("Show Extracted Text"):
                    st.text(page.ocr_text)

            st.markdown("---")

    # ===================================================================
    # V1 — Direct LLM (Single-hop, uses LLMClassifier)
    # ===================================================================
    else:
        from classifier.llm import LLMClassifier

        with st.spinner("🤖 Direct LLM classification (single-hop)..."):
            try:
                # OCR the whole file
                di_client = create_di_client(config)
                ocr_result = asyncio.run(
                    asyncio.to_thread(
                        analyze_document,
                        client=di_client,
                        file_path=tmp_path,
                        model_id=config.azure_di_model,
                    )
                )
                display_images = load_document_images(tmp_path, file_type)

                # Classify each page via LLMClassifier
                classifier = LLMClassifier(config)

                async def _classify_all():
                    results = []
                    for ocr_page in ocr_result.pages:
                        llm_output = await classifier.classify(ocr_page.text)
                        results.append((ocr_page, llm_output))
                    return results

                page_results = asyncio.run(_classify_all())

            except Exception as e:
                log.exception("streamlit_v1_failed", error=str(e))
                st.error(f"Error processing document: {e}")
                st.stop()

        status_msg.success(f"✅ V1 Classification Complete!")
        st.markdown("---")

        for ocr_page, llm_output in page_results:
            page_idx = ocr_page.page_index
            st.subheader(f"Page {page_idx + 1}")
            col_img, col_data = st.columns([1, 1])

            with col_img:
                if page_idx < len(display_images):
                    st.image(display_images[page_idx], use_container_width=True)

            with col_data:
                st.markdown("### 🏷️ Classification")
                primary = llm_output.primary_class.value.upper()
                sub = llm_output.subcategory.value.upper()
                st.success(f"**{primary}** ➔ **{sub}**")

                if llm_output.hospital_name:
                    st.info(f"🏥 **Hospital:** {llm_output.hospital_name}")

                st.metric("OCR Confidence", f"{ocr_page.mean_confidence:.1f}%")

                st.caption("ℹ️ V1 mode — no logprob analysis available")

                with st.expander("Show Extracted Text"):
                    st.text(ocr_page.text)

            st.markdown("---")
