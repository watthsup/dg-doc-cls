import asyncio
import tempfile
from pathlib import Path

import streamlit as st
import structlog

from classifier.llm import LLMClassifier
from config import AppConfig, setup_logging
from graph.builder import build_classification_graph
from graph.state import create_initial_state
from ocr.engine import create_di_client, load_document_images
from pipeline.document import process_document
from pipeline.filesystem import detect_file_type, generate_document_id
from pipeline.graph_adapter import build_logprob_summary, graph_state_to_document_result
from schemas.models import DocumentInput

st.set_page_config(page_title="DocGuru Classifier", layout="wide")

st.title("📄 DocGuru Document Classifier")

st.sidebar.header("Configuration")
max_pages = st.sidebar.number_input("Max Pages to Process", min_value=1, max_value=50, value=10)
pipeline_mode = st.sidebar.radio(
    "Pipeline Mode",
    ["V1 — Direct LLM", "V2 — LangGraph (Hierarchical)"],
    index=0,
)

uploaded_file = st.file_uploader(
    "Upload Medical Document", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    # Save uploaded file to a temporary location so the pipeline can read it
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    # Note: st.empty() allows us to clear this text later if needed
    status_msg = st.empty()
    status_msg.info(f"Preparing {uploaded_file.name}...")

    # Initialize configuration and logging
    config = AppConfig()  # type: ignore[call-arg]
    config = config.model_copy(update={"max_pages": max_pages})
    setup_logging(config)
    log = structlog.get_logger()

    file_type = detect_file_type(tmp_path)

    if not file_type:
        st.error("Unsupported file type!")
        st.stop()

    # -----------------------------------------------------------------------
    # V2 — LangGraph Pipeline
    # -----------------------------------------------------------------------
    if pipeline_mode.startswith("V2"):
        doc_id = generate_document_id(tmp_path)

        with st.spinner("🧠 LangGraph hierarchical classification in progress..."):
            try:
                graph = build_classification_graph(config)
                initial_state = create_initial_state(
                    document_id=doc_id,
                    file_path=str(tmp_path),
                    file_type=file_type,
                )
                thread_config = {"configurable": {"thread_id": doc_id}}
                final_state = graph.invoke(initial_state, config=thread_config)

                display_images = load_document_images(tmp_path, file_type)

                result = graph_state_to_document_result(
                    state=final_state,
                    file_name=uploaded_file.name,
                    total_pages=len(display_images),
                )
            except Exception as e:
                log.exception("streamlit_graph_failed", error=str(e))
                st.error(f"Error processing document: {e}")
                st.stop()

        # HITL Check
        if final_state.get("requires_human_review"):
            status_msg.warning("⚠️ Low confidence — requires human review!")
            st.warning("This document has been flagged for human review. Go to the **Review** page to resolve it.")
        else:
            status_msg.success("✅ Classification Complete!")

        # --- Logprob Analysis Panel ---
        summary = build_logprob_summary(final_state)

        st.markdown("### 🔬 Logprob Analysis (Reliability Engine)")
        col_root, col_sub = st.columns(2)

        with col_root:
            root_margin = summary["root"]["margin"]
            root_color = "green" if root_margin >= config.margin_threshold else "red"
            st.markdown(f"**Root Decision:** `{summary['root']['code']}`")
            st.markdown(f"Margin: :{root_color}[{root_margin:.3f}]")
            st.markdown(f"Confidence: {summary['root']['confidence_pct']:.1f}%")
            if summary["root"]["logprobs"]:
                lp = summary["root"]["logprobs"]
                st.caption(f"Top1: {lp.get('top1_token')} ({lp.get('top1_logprob', 0):.3f})  |  Top2: {lp.get('top2_token')} ({lp.get('top2_logprob', 0):.3f})")

        with col_sub:
            if summary["sub"]["code"]:
                sub_margin = summary["sub"]["margin"]
                sub_color = "green" if sub_margin >= config.margin_threshold else "red"
                st.markdown(f"**Sub Decision:** `{summary['sub']['code']}`")
                st.markdown(f"Margin: :{sub_color}[{sub_margin:.3f}]")
                st.markdown(f"Confidence: {summary['sub']['confidence_pct']:.1f}%")
                if summary["sub"]["logprobs"]:
                    lp = summary["sub"]["logprobs"]
                    st.caption(f"Top1: {lp.get('top1_token')} ({lp.get('top1_logprob', 0):.3f})  |  Top2: {lp.get('top2_token')} ({lp.get('top2_logprob', 0):.3f})")
            else:
                st.info("Sub-classification not reached (interrupted at root level)")

        st.caption(f"Execution Trail: {' → '.join(summary['execution_trail'])}")

        if result.hospital_name:
            st.info(f"🏥 **Document Hospital:** {result.hospital_name}")

        # --- Page Display (same as V1) ---
        st.markdown("---")
        for page in result.pages:
            st.subheader(f"Page {page.page_index + 1}")
            col_img, col_data = st.columns([1, 1])
            with col_img:
                if page.page_index < len(display_images):
                    st.image(display_images[page.page_index], use_container_width=True)
            with col_data:
                st.markdown("### 🏷️ Classification")
                primary = page.classification.primary_class.value.upper()
                sub = page.classification.subcategory.value.upper()
                st.success(f"**{primary}** ➔ **{sub}**")
                if page.classification.hospital_name:
                    st.info(f"🏥 **Hospital:** {page.classification.hospital_name}")
                with st.expander("Show Extracted Text"):
                    st.text(page.ocr_text)
            st.markdown("---")

    # -----------------------------------------------------------------------
    # V1 — Direct LLM Pipeline (original)
    # -----------------------------------------------------------------------
    else:
        doc_input = DocumentInput(
            document_id=generate_document_id(tmp_path),
            file_path=tmp_path,
            file_type=file_type,  # type: ignore[arg-type]
        )

        with st.spinner("AI is classifying the document... this may take a moment."):
            # Initialize pipeline components
            classifier = LLMClassifier(config)
            di_client = create_di_client(config)

            try:
                # Run the asynchronous processing pipeline
                result = asyncio.run(
                    process_document(
                        document=doc_input,
                        classifier=classifier,
                        config=config,
                        di_client=di_client,
                    )
                )

                # Load images for rendering in the UI
                display_images = load_document_images(tmp_path, file_type)

            except Exception as e:
                log.exception("streamlit_processing_failed", error=str(e))
                st.error(f"Error processing document: {e}")
                st.stop()

        status_msg.success(f"Classification Complete! ({result.processing_metadata.processing_time_ms} ms)")

        if result.hospital_name:
            st.info(f"🏥 **Document Hospital (Majority Vote):** {result.hospital_name}")

        # --- UI Layout ---
        st.markdown("---")

        for page in result.pages:
            st.subheader(f"Page {page.page_index + 1}")
            
            # Two-column layout
            col_img, col_data = st.columns([1, 1])

            with col_img:
                if page.page_index < len(display_images):
                    # use_container_width replaces use_column_width in newer Streamlit versions
                    st.image(display_images[page.page_index], use_container_width=True)
                else:
                    st.warning("Image rendering not available for this page.")

            with col_data:
                # Classification Results
                st.markdown("### 🏷️ Classification")
                primary = page.classification.primary_class.value.upper()
                sub = page.classification.subcategory.value.upper()
                
                st.success(f"**{primary}** ➔ **{sub}**")

                if page.classification.hospital_name:
                    st.info(f"🏥 **Hospital:** {page.classification.hospital_name}")

                # Overall Confidence Bar
                st.progress(page.confidence, text=f"Overall Confidence: {page.confidence:.1%}")

                # Signal Scores
                st.markdown("### 📊 Signals")
                col_sig1, col_sig2 = st.columns(2)
                col_sig1.metric("OCR Confidence", f"{page.signals.ocr_confidence:.1%}")
                col_sig2.metric("Image Quality Score", f"{page.signals.quality_score:.1%}")

                st.markdown("#### Image Quality Sub-Scores")
                col_q1, col_q2, col_q3 = st.columns(3)
                
                blur = page.quality_assessment.blur_score
                contrast = page.quality_assessment.contrast_score
                skew = page.quality_assessment.skew_angle
                
                def get_color(val: float, thresh: float, is_skew: bool = False) -> str:
                    if is_skew:
                        return "red" if abs(val) > thresh else "green"
                    return "red" if val < thresh else "green"
                    
                blur_color = get_color(blur, 0.3)
                contrast_color = get_color(contrast, 0.3)
                skew_color = get_color(skew, 5.0, is_skew=True)
                
                col_q1.markdown(f"**Blur:** :{blur_color}[{blur:.2f}]")
                col_q2.markdown(f"**Contrast:** :{contrast_color}[{contrast:.2f}]")
                col_q3.markdown(f"**Skew:** :{skew_color}[{skew:.1f}°]")

                # Quality Issues Warnings
                if page.quality_assessment.issues:
                    st.markdown("### ⚠️ Quality Issues")
                    for issue in page.quality_assessment.issues:
                        st.warning(issue)

                # OCR Text Toggle
                with st.expander("Show Extracted Text"):
                    st.text(page.ocr_text)

            st.markdown("---")
