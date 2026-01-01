import streamlit as st
import pandas as pd
import json
import io
import os
from google.cloud import storage
from google.oauth2 import service_account
import plotly.express as px
from PIL import Image, ImageOps

# ==========================================
# CONFIGURATION
# ==========================================
BUCKET_NAME = 'venice_singlepages_37' 
BUCKET_PREFIX = "" 
CSV_FILENAME = 'aggregated_results.csv' 
GITHUB_RAW_URL = "https://raw.githubusercontent.com/sets-architecture/venice-verifier/refs/heads/main/aggregated_annotations_claude.csv"

st.set_page_config(layout="wide", page_title="Venice Verifier")

# ==========================================
# AUTHENTICATION
# ==========================================
st.sidebar.title("Configuration")
uploaded_key = st.sidebar.file_uploader("1. Upload GCS JSON Key", type='json')

if not uploaded_key:
    st.info("Please upload your Google Cloud JSON key to authenticate.")
    st.stop()

try:
    key_data = json.load(uploaded_key)
    credentials = service_account.Credentials.from_service_account_info(key_data)
    client = storage.Client(credentials=credentials)
except Exception as e:
    st.error(f"Invalid Key File: {e}")
    st.stop()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_and_prep_dataframe(df):
    if 'page_id' in df.columns:
        df.rename(columns={'page_id': 'Page_ID'}, inplace=True)
    if 'Verification_Status' not in df.columns:
        df.insert(0, 'Verification_Status', False)
    if 'Notes' not in df.columns:
        df.insert(1, 'Notes', "")
    for col in df.columns:
        if col != 'Verification_Status':
            df[col] = df[col].fillna("").astype(str)
    df['Verification_Status'] = df['Verification_Status'].replace({'True': True, 'False': False}).astype(bool)
    return df

def load_csv_from_gcs():
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        if blob.exists():
            data = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(data))
            return clean_and_prep_dataframe(df)
    except Exception as e:
        st.error(f"Error loading CSV from Cloud: {e}")
    return None

def load_csv_from_github():
    try:
        df = pd.read_csv(GITHUB_RAW_URL)
        return clean_and_prep_dataframe(df)
    except Exception as e:
        st.error(f"Error reading from GitHub: {e}")
        return None

def save_csv_to_gcs(df):
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        blob.upload_from_string(df.to_csv(index=False), 'text/csv')
        st.toast(f"✅ Saved {CSV_FILENAME} to Cloud!", icon="☁️")
    except Exception as e:
        st.error(f"Failed to save to Cloud: {e}")

@st.cache_data(show_spinner=False)
def load_image_from_gcs(blob_name):
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        image_data = blob.download_as_bytes()
        return io.BytesIO(image_data)
    except Exception as e:
        return None

# ==========================================
# SIDEBAR: DATA SYNC
# ==========================================
st.sidebar.subheader("Data Sync")
if st.sidebar.button("⚠️ Force Update from GitHub"):
    with st.spinner("Pulling fresh data from GitHub..."):
        df_git = load_csv_from_github()
        if df_git is not None:
            st.session_state['df'] = df_git
            st.session_state.page_index = 0
            save_csv_to_gcs(df_git)
            st.success("Data updated from GitHub and saved to Cloud!")
            st.rerun()

# ==========================================
# APP STATE
# ==========================================
if 'df' not in st.session_state:
    df = load_csv_from_gcs()
    if df is None:
        if os.path.exists(CSV_FILENAME):
            df = pd.read_csv(CSV_FILENAME)
            df = clean_and_prep_dataframe(df)
            st.toast("Loaded data from local disk.")
        else:
            st.warning("No data found. Attempting GitHub pull...")
            df = load_csv_from_github()
            if df is not None:
                save_csv_to_gcs(df)
    
    if df is None:
        st.error("Could not load data.")
        st.stop()

    st.session_state['df'] = df
    st.session_state.page_index = 0
    st.session_state.rotation = 0

df = st.session_state['df']
unique_pages = df['Page_ID'].unique()

# ==========================================
# NAVIGATION
# ==========================================
def reset_rotation():
    st.session_state.rotation = 0

def next_page():
    if st.session_state.page_index < len(unique_pages) - 1:
        st.session_state.page_index += 1
        reset_rotation()
def prev_page():
    if st.session_state.page_index > 0:
        st.session_state.page_index -= 1
        reset_rotation()

st.sidebar.write("---")
c1, c2 = st.sidebar.columns(2)
c1.button("⬅️ Previous", on_click=prev_page)
c2.button("Next ➡️", on_click=next_page)

selected_page = st.sidebar.selectbox(
    "Jump to Image", unique_pages, index=st.session_state.page_index
)

if selected_page != unique_pages[st.session_state.page_index]:
    st.session_state.page_index = list(unique_pages).index(selected_page)
    reset_rotation()

# ==========================================
# MAIN INTERFACE
# ==========================================
col_img, col_data = st.columns([1.2, 0.8])

# --- LEFT: ZOOMABLE IMAGE ---
with col_img:
    st.subheader(f"📄 {selected_page}")
    if st.button("⟳ Rotate 90°"):
        st.session_state.rotation = (st.session_state.rotation - 90) % 360
        st.rerun()

    clean_name = selected_page.replace("gs://", "").split("/")[-1]
    blob_path = f"{BUCKET_PREFIX}/{clean_name}".replace("//", "/")
    if blob_path.startswith("/"): blob_path = blob_path[1:]
    
    img_bytes = load_image_from_gcs(blob_path)
    
    if img_bytes:
        pil_image = Image.open(img_bytes)
        pil_image = ImageOps.exif_transpose(pil_image)
        if st.session_state.rotation != 0:
            pil_image = pil_image.rotate(st.session_state.rotation, expand=True)
        
        fig = px.imshow(pil_image)
        fig.update_layout(
            width=None, height=800, margin=dict(l=0, r=0, b=0, t=0),
            xaxis={'visible': False}, yaxis={'visible': False},
            dragmode='pan', hovermode=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 Scroll to Zoom | Click & Drag to Pan | Double-click to Reset")
    else:
        st.error(f"Image not found: {blob_path}")

# --- RIGHT: VERTICAL EDITOR ---
with col_data:
    st.subheader("📝 Verification")
    
    page_mask = df['Page_ID'] == selected_page
    row_data = df.loc[page_mask]

    # Metadata
    verified_val = row_data['Verification_Status'].values[0]
    notes_val = row_data['Notes'].values[0]
    new_verified = st.checkbox("✅ Mark Verified", value=bool(verified_val))
    new_notes = st.text_area("Notes", value=str(notes_val))

    if new_verified != verified_val or new_notes != notes_val:
        df.loc[page_mask, 'Verification_Status'] = new_verified
        df.loc[page_mask, 'Notes'] = new_notes
        st.session_state['df'] = df

    st.divider()

    # Prep Data
    cols_to_exclude = ['Page_ID', 'Verification_Status', 'Notes']
    editable_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    vertical_df = row_data[editable_cols].T
    vertical_df.columns = ['Value']
    vertical_df.index.name = 'Field'
    vertical_df = vertical_df.reset_index()

    # --- VIEW MODE TOGGLE ---
    view_mode = st.radio("Editor View:", ["Compact Grid", "Expanded Form (Word Wrap)"], horizontal=True)

    if view_mode == "Compact Grid":
        # GRID MODE (Fast, Compact)
        edited_vertical = st.data_editor(
            vertical_df,
            key="v_editor",
            height=650, 
            hide_index=True,
            width="stretch",
            column_config={
                "Field": st.column_config.TextColumn("Field", width="medium", disabled=True),
                "Value": st.column_config.TextColumn("Value", width="large")
            }
        )
        # Save Grid Changes
        if not edited_vertical.equals(vertical_df):
            updated_values = dict(zip(edited_vertical['Field'], edited_vertical['Value']))
            for col, val in updated_values.items():
                df.loc[page_mask, col] = val
            st.session_state['df'] = df

    else:
        # FORM MODE (Full Word Wrap)
        with st.form("expanded_form"):
            new_values = {}
            for i, row in vertical_df.iterrows():
                # Text Area allows full word wrap and adjustable height
                label_txt = row['Field']
                curr_val = row['Value']
                # Make text area taller if content is long
                h = 100 if len(curr_val) > 50 else None 
                
                new_val = st.text_area(label=label_txt, value=curr_val, height=h)
                new_values[label_txt] = new_val
            
            if st.form_submit_button("💾 Save Form Changes"):
                for col, val in new_values.items():
                    df.loc[page_mask, col] = val
                st.session_state['df'] = df
                st.toast("Changes applied locally. Click 'Save Changes to Cloud' to persist.")
                st.rerun()

    st.write("---")
    if st.button("☁️ Save Changes to Cloud", type="primary"):
        save_csv_to_gcs(df)

    verified_count = df['Verification_Status'].sum()
    st.progress(verified_count / len(df))
    st.caption(f"Verified: {verified_count} / {len(df)}")