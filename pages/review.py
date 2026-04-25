"""HITL Review Page — Human-in-the-Loop adjudication for uncertain documents.

Lists documents that have been flagged by the LangGraph pipeline
for human review. The operator can see the AI conflict (Top 1 vs Top 2)
and select the correct classification code to resume the pipeline.
"""

import sqlite3
from pathlib import Path

import streamlit as st

from config import AppConfig
from graph.builder import build_classification_graph
from schemas.models import (
    CODE_TO_PRIMARY,
    CODE_TO_SUBCATEGORY,
    VALID_MED_SUB_CODES,
    VALID_NONMED_SUB_CODES,
    VALID_ROOT_CODES,
    ClassificationCode,
)

st.set_page_config(page_title="DocGuru — Human Review", layout="wide")
st.title("🧑‍⚕️ Human-in-the-Loop Review")
st.markdown("Documents flagged by the AI for human adjudication are listed below.")

config = AppConfig()  # type: ignore[call-arg]

checkpoint_path = Path(config.checkpoint_db_path)

if not checkpoint_path.exists():
    st.info("No checkpoint database found. Process documents using the V2 pipeline first.")
    st.stop()

# Query the checkpoint DB for pending reviews
try:
    conn = sqlite3.connect(str(checkpoint_path))
    cursor = conn.cursor()

    # LangGraph stores checkpoints with thread_id metadata
    cursor.execute("""
        SELECT DISTINCT thread_id 
        FROM checkpoints 
        ORDER BY thread_id
    """)
    thread_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
except Exception as e:
    st.error(f"Error reading checkpoint database: {e}")
    st.stop()

if not thread_ids:
    st.info("✅ No documents pending review. All clear!")
    st.stop()

st.markdown(f"**{len(thread_ids)}** document(s) in checkpoint database.")

# Build graph for state inspection
graph = build_classification_graph(config)

for thread_id in thread_ids:
    thread_config = {"configurable": {"thread_id": thread_id}}

    try:
        state_snapshot = graph.get_state(thread_config)
        state = state_snapshot.values
    except Exception:
        continue

    if not state:
        continue

    # Only show documents that need review
    requires_review = state.get("requires_human_review", False)
    is_uncertain = state.get("is_uncertain", False)

    with st.expander(
        f"📄 {state.get('document_id', thread_id)} — "
        f"{'⚠️ NEEDS REVIEW' if requires_review or is_uncertain else '✅ Complete'}",
        expanded=requires_review or is_uncertain,
    ):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### AI Classification")
            st.markdown(f"**Root Code:** `{state.get('root_code', 'N/A')}`")
            st.markdown(f"**Sub Code:** `{state.get('sub_code', 'N/A')}`")

            st.markdown("#### Margin Analysis")
            root_margin = state.get("root_margin", 0.0)
            sub_margin = state.get("sub_margin", 0.0)
            uncertainty_stage = state.get("uncertainty_stage", "N/A")

            root_color = "green" if root_margin >= config.margin_threshold else "red"
            sub_color = "green" if sub_margin >= config.margin_threshold else "red"

            st.markdown(f"Root Margin: :{root_color}[{root_margin:.3f}]")
            st.markdown(f"Sub Margin: :{sub_color}[{sub_margin:.3f}]")
            st.markdown(f"Uncertain Stage: **{uncertainty_stage}**")

            # Show logprob conflict
            st.markdown("#### AI Conflict")
            root_lp = state.get("root_logprobs", {})
            sub_lp = state.get("sub_logprobs", {})
            if root_lp:
                st.caption(f"Root — Top1: {root_lp.get('top1_token')} vs Top2: {root_lp.get('top2_token')}")
            if sub_lp:
                st.caption(f"Sub — Top1: {sub_lp.get('top1_token')} vs Top2: {sub_lp.get('top2_token')}")

            st.markdown("#### Execution Trail")
            trail = state.get("execution_trail", [])
            st.caption(" → ".join(trail) if trail else "No trail")

        with col2:
            st.markdown("#### OCR Text Preview")
            ocr_text = state.get("azure_ocr_text", "")
            st.text_area(
                "Extracted Text",
                value=ocr_text[:2000] if ocr_text else "No text available",
                height=300,
                key=f"text_{thread_id}",
                disabled=True,
            )

        # Human override controls
        if requires_review or is_uncertain:
            st.markdown("---")
            st.markdown("#### 🔧 Human Override")

            # Determine which codes to offer based on uncertainty stage
            if uncertainty_stage == "root":
                code_options = [c.value for c in VALID_ROOT_CODES]
                label = "Select correct ROOT classification:"
            else:
                root_code = state.get("root_code", "MED")
                if root_code == "MED":
                    code_options = [c.value for c in VALID_MED_SUB_CODES]
                else:
                    code_options = [c.value for c in VALID_NONMED_SUB_CODES]
                label = "Select correct SUB classification:"

            selected_code = st.selectbox(
                label,
                code_options,
                key=f"override_{thread_id}",
            )

            if st.button(f"✅ Submit Override for {thread_id}", key=f"btn_{thread_id}"):
                try:
                    # Resume graph with human override
                    if uncertainty_stage == "root":
                        override_state = {
                            "root_code": selected_code,
                            "is_uncertain": False,
                            "requires_human_review": False,
                            "human_override_code": selected_code,
                        }
                    else:
                        override_state = {
                            "sub_code": selected_code,
                            "is_uncertain": False,
                            "requires_human_review": False,
                            "human_override_code": selected_code,
                        }

                    graph.update_state(thread_config, override_state)
                    # Resume execution
                    graph.invoke(None, config=thread_config)

                    st.success(f"✅ Document {thread_id} resolved with code: {selected_code}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resuming pipeline: {e}")
