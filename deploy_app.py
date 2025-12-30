import streamlit as st
import pandas as pd
import json
import io
import os
from google.cloud import storage
from google.oauth2 import service_account
import plotly.express as px
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
BUCKET_NAME = 'venice_singlepages_37' 
BUCKET_PREFIX = "" 
CSV_FILENAME = 'aggregated_results.csv' 

st.set_page_config(layout="wide", page_title="Venice Verifier")

# ==========================================
# AUTHENTICATION
# ==========================================
st.sidebar.title("Login")
uploaded_key = st.sidebar.file_uploader("Upload Service Account JSON", type='json')

if not uploaded_key:
    st.warning("Please upload a Google Cloud JSON key to proceed.")
    st.stop()

# Initialize GCS Client
key_data = json.load(uploaded_key)
credentials = service_account.Credentials.from_service_account_info(key_data)
client = storage.Client(credentials=credentials)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_csv_from_gcs():
    """Loads the main data CSV from GCS."""
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        if blob.exists():
            data = blob.download_as_bytes()
            return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        st.error(f"Error loading CSV from Cloud: {e}")
    return None

def save_csv_to_gcs(df):
    """Saves the DataFrame back to GCS."""
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        blob.upload_from_string(df.to_csv(index=False), 'text/csv')
        st.toast(f"✅ Saved {CSV_FILENAME} to Cloud!", icon="☁️")
    except Exception as e:
        st.error(f"Failed to save to Cloud: {e}")

@st.cache_data(show_spinner=False)
def load_image_from_gcs(blob_name):
    """Downloads image bytes from GCS."""
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        image_data = blob.download_as_bytes()
        return io.BytesIO(image_data)
    except Exception as e:
        return None

# ==========================================
# APP STATE INITIALIZATION
# ==========================================
if 'df' not in st.session_state:
    df = load_csv_from_gcs()
    
    # Fallback for local testing
    if df is None:
        if os.path.exists(CSV_FILENAME):
            df = pd.read_csv(CSV_FILENAME)
        else:
            st.error("No CSV found. Please upload 'aggregated_results.csv' to your bucket.")
            st.stop()

    # --- CRITICAL FIX: TYPE CONVERSION ---
    # Ensure Notes and Text columns are Strings, not Floats (NaN)
    
    # 1. Normalize Page_ID
    if 'page_id' in df.columns:
        df.rename(columns={'page_id': 'Page_ID'}, inplace=True)

    # 2. Ensure Metadata columns exist
    if 'Verification_Status' not in df.columns:
        df.insert(0, 'Verification_Status', False)
    if 'Notes' not in df.columns:
        df.insert(1, 'Notes', "")

    # 3. FORCE STRING TYPES (Fixes the crash)
    # Convert 'Notes' explicitly to string, replacing NaNs with empty string
    df['Notes'] = df['Notes'].fillna("").astype(str)
    
    # Convert any other object columns to string to be safe
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("").astype(str)

    st.session_state['df'] = df
    st.session_state.page_index = 0

df = st.session_state['df']
unique_pages = df['Page_ID'].unique()

# ==========================================
# NAVIGATION
# ==========================================
def next_page():
    if st.session_state.page_index < len(unique_pages) - 1:
        st.session_state.page_index += 1
def prev_page():
    if st.session_state.page_index > 0:
        st.session_state.page_index -= 1

st.sidebar.write("---")
c1, c2 = st.sidebar.columns(2)
c1.button("⬅️ Previous", on_click=prev_page)
c2.button("Next ➡️", on_click=next_page)

# Dropdown (Synced with Index)
selected_page = st.sidebar.selectbox(
    "Jump to Image", 
    unique_pages, 
    index=st.session_state.page_index
)

if selected_page != unique_pages[st.session_state.page_index]:
    st.session_state.page_index = list(unique_pages).index(selected_page)

# ==========================================
# MAIN INTERFACE
# ==========================================
col_img, col_data = st.columns([1.2, 0.8])

# --- LEFT COLUMN: ZOOMABLE IMAGE ---
with col_img:
    st.subheader(f"📄 {selected_page}")
    
    clean_name = selected_page.replace("gs://", "").split("/")[-1]
    blob_path = f"{BUCKET_PREFIX}/{clean_name}".replace("//", "/")
    if blob_path.startswith("/"): blob_path = blob_path[1:]
    
    img_bytes = load_image_from_gcs(blob_path)
    
    if img_bytes:
        pil_image = Image.open(img_bytes)
        
        # Plotly Figure for Zoom/Pan
        fig = px.imshow(pil_image)
        fig.update_layout(
            width=None, 
            height=800, 
            margin=dict(l=0, r=0, b=0, t=0),
            xaxis={'visible': False, 'showticklabels': False},
            yaxis={'visible': False, 'showticklabels': False},
            dragmode='pan',
            hovermode=False
        )
        # Use st.plotly_chart (use_container_width is still correct here)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 Scroll to Zoom | Click & Drag to Pan | Double-click to Reset")
    else:
        st.error(f"Image not found: {blob_path}")

# --- RIGHT COLUMN: VERTICAL DATA EDITOR ---
with col_data:
    st.subheader("📝 Verification")
    
    page_mask = df['Page_ID'] == selected_page
    row_data = df.loc[page_mask]

    # 1. Metadata Controls
    verified_val = row_data['Verification_Status'].values[0]
    notes_val = row_data['Notes'].values[0]

    new_verified = st.checkbox("✅ Mark Page as Verified", value=bool(verified_val))
    new_notes = st.text_area("Notes / Issues", value=str(notes_val))

    # Update Metadata
    if new_verified != verified_val or new_notes != notes_val:
        df.loc[page_mask, 'Verification_Status'] = new_verified
        df.loc[page_mask, 'Notes'] = new_notes
        st.session_state['df'] = df

    st.divider()

    # 2. Transpose for Vertical List
    cols_to_exclude = ['Page_ID', 'Verification_Status', 'Notes']
    editable_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    vertical_df = row_data[editable_cols].T
    vertical_df.columns = ['Value']
    vertical_df.index.name = 'Field'
    vertical_df = vertical_df.reset_index()

    # 3. Render Editor (Using width="stretch" to fix warning)
    edited_vertical = st.data_editor(
        vertical_df,
        key="v_editor",
        height=650, 
        hide_index=True,
        width="stretch", # Replaces use_container_width=True
        column_config={
            "Field": st.column_config.TextColumn("Field", disabled=True),
            "Value": st.column_config.TextColumn("Value") 
        }
    )

    # 4. Save List Changes
    if not edited_vertical.equals(vertical_df):
        updated_values = dict(zip(edited_vertical['Field'], edited_vertical['Value']))
        for col, val in updated_values.items():
            df.loc[page_mask, col] = val
        st.session_state['df'] = df

    # 5. Global Save Button
    st.write("---")
    if st.button("💾 Save Changes to Cloud", type="primary"):
        save_csv_to_gcs(df)

    # Progress Bar
    verified_count = df['Verification_Status'].sum()
    total = len(df)
    st.progress(verified_count / total)
    st.caption(f"Verified: {verified_count} / {total}")
