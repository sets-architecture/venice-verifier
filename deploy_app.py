import streamlit as st
import pandas as pd
import json
import io
import os
from google.cloud import storage
from google.oauth2 import service_account
import plotly.express as px
from PIL import Image, ImageOps  # <--- NEW IMPORT

# ==========================================
# CONFIGURATION
# ==========================================
BUCKET_NAME = 'venice_singlepages_37' 
BUCKET_PREFIX = "" 
CSV_FILENAME = 'aggregated_annotations_claude.csv' 

st.set_page_config(layout="wide", page_title="Venice Verifier")

# ==========================================
# AUTHENTICATION
# ==========================================
st.sidebar.title("Login")
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
# HELPERS
# ==========================================
def load_csv_from_gcs():
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        if blob.exists():
            data = blob.download_as_bytes()
            return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
    return None

def save_csv_to_gcs(df):
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        blob.upload_from_string(df.to_csv(index=False), 'text/csv')
        st.toast(f"✅ Saved {CSV_FILENAME} to Cloud!", icon="☁️")
    except Exception as e:
        st.error(f"Failed to save: {e}")

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
# APP STATE
# ==========================================
if 'df' not in st.session_state:
    df = load_csv_from_gcs()
    
    # Local fallback
    if df is None:
        if os.path.exists(CSV_FILENAME):
            df = pd.read_csv(CSV_FILENAME)
            st.toast("Loaded from local disk.", icon="📂")
    
    # Manual Upload fallback
    if df is None:
        st.warning(f"Could not find data. Please upload CSV.")
        uploaded_csv = st.file_uploader("2. Upload Data CSV", type='csv')
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            save_csv_to_gcs(df)
            st.rerun()
        else:
            st.stop()

    # Normalize Columns
    if 'page_id' in df.columns:
        df.rename(columns={'page_id': 'Page_ID'}, inplace=True)
    if 'Verification_Status' not in df.columns:
        df.insert(0, 'Verification_Status', False)
    if 'Notes' not in df.columns:
        df.insert(1, 'Notes', "")

    # Clean Data Types (Strings)
    for col in df.columns:
        if col != 'Verification_Status':
            df[col] = df[col].fillna("").astype(str)
            
    df['Verification_Status'] = df['Verification_Status'].replace({'True': True, 'False': False}).astype(bool)

    st.session_state['df'] = df
    st.session_state.page_index = 0
    st.session_state.rotation = 0 # <--- NEW: Initialize Rotation

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
        reset_rotation() # Reset rotation on page change
def prev_page():
    if st.session_state.page_index > 0:
        st.session_state.page_index -= 1
        reset_rotation()

st.sidebar.write("---")
c1, c2 = st.sidebar.columns(2)
c1.button("⬅️ Previous", on_click=prev_page)
c2.button("Next ➡️", on_click=next_page)

selected_page = st.sidebar.selectbox(
    "Jump to Image", 
    unique_pages, 
    index=st.session_state.page_index
)

# Sync Index and Reset Rotation if Dropdown changed
if selected_page != unique_pages[st.session_state.page_index]:
    st.session_state.page_index = list(unique_pages).index(selected_page)
    reset_rotation()

# ==========================================
# MAIN INTERFACE
# ==========================================
col_img, col_data = st.columns([1.2, 0.8])

# --- LEFT: ZOOMABLE IMAGE (WITH ROTATION) ---
with col_img:
    st.subheader(f"📄 {selected_page}")
    
    # 1. Rotate Button
    if st.button("⟳ Rotate 90°"):
        st.session_state.rotation = (st.session_state.rotation - 90) % 360
        st.rerun()

    clean_name = selected_page.replace("gs://", "").split("/")[-1]
    blob_path = f"{BUCKET_PREFIX}/{clean_name}".replace("//", "/")
    if blob_path.startswith("/"): blob_path = blob_path[1:]
    
    img_bytes = load_image_from_gcs(blob_path)
    
    if img_bytes:
        pil_image = Image.open(img_bytes)
        
        # --- FIX 1: Auto-Fix Orientation Metadata ---
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # --- FIX 2: Apply Manual Rotation ---
        if st.session_state.rotation != 0:
            pil_image = pil_image.rotate(st.session_state.rotation, expand=True)
        
        # Display Plotly
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

    verified_val = row_data['Verification_Status'].values[0]
    notes_val = row_data['Notes'].values[0]

    new_verified = st.checkbox("✅ Mark Verified", value=bool(verified_val))
    new_notes = st.text_area("Notes", value=str(notes_val))

    if new_verified != verified_val or new_notes != notes_val:
        df.loc[page_mask, 'Verification_Status'] = new_verified
        df.loc[page_mask, 'Notes'] = new_notes
        st.session_state['df'] = df

    st.divider()

    cols_to_exclude = ['Page_ID', 'Verification_Status', 'Notes']
    editable_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    vertical_df = row_data[editable_cols].T
    vertical_df.columns = ['Value']
    vertical_df.index.name = 'Field'
    vertical_df = vertical_df.reset_index()

    edited_vertical = st.data_editor(
        vertical_df,
        key="v_editor",
        height=650, 
        hide_index=True,
        width="stretch",
        column_config={
            "Field": st.column_config.TextColumn("Field", disabled=True),
            "Value": st.column_config.TextColumn("Value") 
        }
    )

    if not edited_vertical.equals(vertical_df):
        updated_values = dict(zip(edited_vertical['Field'], edited_vertical['Value']))
        for col, val in updated_values.items():
            df.loc[page_mask, col] = val
        st.session_state['df'] = df

    st.write("---")
    if st.button("💾 Save Changes to Cloud", type="primary"):
        save_csv_to_gcs(df)

    verified_count = df['Verification_Status'].sum()
    st.progress(verified_count / len(df))
    st.caption(f"Verified: {verified_count} / {len(df)}")