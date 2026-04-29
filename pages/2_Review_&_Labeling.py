import streamlit as st
import pandas as pd
from pathlib import Path
import os

st.set_page_config(page_title="Review & Labeling", page_icon="📝", layout="wide")

st.title("📝 Ground Truth Labeling & Review")

# --- 1. Sidebar Configuration ---
st.sidebar.header("Configuration")
csv_path = st.sidebar.text_input("Path to batch_results.csv", value="./output/batch_results.csv")
root_img_dir = st.sidebar.text_input("Root Image Directory", value="./output/images")

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Ensure review columns exist
        if 'reviewed_root_code' not in df.columns:
            df['reviewed_root_code'] = df['root_code']
        if 'reviewed_sub_code' not in df.columns:
            df['reviewed_sub_code'] = df['sub_code']
        if 'is_corrected' not in df.columns:
            df['is_corrected'] = False
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def save_data():
    if st.session_state.df is not None and not st.session_state.df.empty:
        save_path = Path(csv_path).parent / "batch_results_reviewed.csv"
        st.session_state.df.to_csv(save_path, index=False)
        st.toast(f"Progress saved to {save_path.name}", icon="✅")

if st.sidebar.button("Load Data"):
    st.session_state.df = load_data(csv_path)
    st.session_state.current_idx = 0
    st.toast("Data loaded successfully!")

if "df" not in st.session_state or st.session_state.df.empty:
    st.info("Please load a valid CSV file from the sidebar to begin.")
    st.stop()

df = st.session_state.df
idx = st.session_state.current_idx
total_rows = len(df)

if idx >= total_rows:
    st.success("🎉 You have reached the end of the dataset!")
    st.stop()

row = df.iloc[idx]

# --- Helper logic for state ---
def next_item():
    if st.session_state.current_idx < total_rows - 1:
        st.session_state.current_idx += 1

def prev_item():
    if st.session_state.current_idx > 0:
        st.session_state.current_idx -= 1

def next_uncertain():
    # Find next row where is_uncertain is True
    for i in range(idx + 1, total_rows):
        if str(df.iloc[i]['is_uncertain']).lower() == 'true':
            st.session_state.current_idx = i
            return
    st.warning("No more uncertain items found ahead.")

def mark_reviewed(new_root: str, new_sub: str):
    is_changed = (str(row['root_code']) != new_root) or (str(row['sub_code']) != new_sub)
    st.session_state.df.at[idx, 'reviewed_root_code'] = new_root
    st.session_state.df.at[idx, 'reviewed_sub_code'] = new_sub
    st.session_state.df.at[idx, 'is_corrected'] = is_changed
    save_data()
    next_item()

# --- 2. Main Layout ---
st.progress((idx + 1) / total_rows, text=f"Progress: {idx + 1} / {total_rows}")

col_img, col_act = st.columns([1.2, 1])

# --- Left Column: Image & Context ---
with col_img:
    st.subheader(f"📄 {row['file_name']} (Page {row['page_index']})")
    
    # Prediction Header
    is_uncertain = str(row['is_uncertain']).lower() == 'true'
    msg_color = "red" if is_uncertain else "green"
    st.markdown(f"### Current Prediction: :{msg_color}[{row['root_code']} ➔ {row['sub_code']}]")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Root Score", f"{row['root_score']}")
    c2.metric("Margin", f"{row['root_margin']} / {row['sub_margin']}")
    c3.metric("Confidence", f"{row['root_conf_pct']}% / {row['sub_conf_pct']}%")
    
    # Image loading
    try:
        base_name = Path(str(row['file_name'])).stem
        page_idx = int(row['page_index'])
        # Path format: root_dir / root_code / sub_code / base_name_p_page_idx.png
        # Use the CURRENT prediction to locate the file, not the reviewed one, assuming files are sorted by initial prediction
        img_file = Path(root_img_dir) / str(row['root_code']) / str(row['sub_code']) / f"{base_name}_p_{page_idx}.png"
        
        if img_file.exists():
            st.image(str(img_file), use_column_width=True)
        else:
            st.warning(f"Image not found at expected path:\n`{img_file}`")
    except Exception as e:
        st.error(f"Could not construct image path: {e}")
        
    with st.expander("View OCR Text"):
        st.text(row['ocr_text'] if 'ocr_text' in row else "No OCR text available")
        st.caption(f"Execution Trail: {row['trail']}")

# --- Right Column: Actions & Labeling ---
with col_act:
    st.subheader("🛠️ Review Actions")
    
    # Navigation
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        st.button("⬅️ Prev", on_click=prev_item, use_container_width=True, disabled=(idx == 0))
    with nav2:
        # Mark as correct (no change)
        st.button("✅ Confirm", on_click=mark_reviewed, args=(row['root_code'], row['sub_code']), use_container_width=True, type="primary")
    with nav3:
        st.button("Next ➡️", on_click=next_item, use_container_width=True, disabled=(idx == total_rows - 1))
    with nav4:
        st.button("🔍 Next Uncertain", on_click=next_uncertain, use_container_width=True)

    st.divider()
    st.markdown("### 🚑 Medical (MED)")
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    
    with m1:
        if st.button("LAB", use_container_width=True): mark_reviewed("MED", "LAB")
    with m2:
        if st.button("HEALTH CHECK", use_container_width=True): mark_reviewed("MED", "CHK")
    with m3:
        if st.button("IPD/OPD DOC", use_container_width=True): mark_reviewed("MED", "IPD_OPD_DOCUMENT")
    with m4:
        if st.button("OTHER (MED)", use_container_width=True): mark_reviewed("MED", "MEDICAL_OTHER")
        
    st.divider()
    st.markdown("### 🪪 Non-Medical (NON)")
    n1, n2 = st.columns(2)
    n3, n4 = st.columns(2)
    
    with n1:
        if st.button("ID", use_container_width=True): mark_reviewed("NON", "ID")
    with n2:
        if st.button("PASSPORT", use_container_width=True): mark_reviewed("NON", "PAS")
    with n3:
        if st.button("FINANCIAL", use_container_width=True): mark_reviewed("NON", "FIN")
    with n4:
        if st.button("OTHER (NON)", use_container_width=True): mark_reviewed("NON", "OTH")

    st.divider()
    
    # Status showing what is currently recorded
    is_corr = row.get('is_corrected', False)
    rev_r = row.get('reviewed_root_code', row['root_code'])
    rev_s = row.get('reviewed_sub_code', row['sub_code'])
    
    if is_corr:
        st.info(f"**Status:** Corrected to `{rev_r} ➔ {rev_s}`")
    else:
        st.success(f"**Status:** Accepted as `{rev_r} ➔ {rev_s}`")
