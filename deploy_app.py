import streamlit as st
import pandas as pd
import json
import io
from google.cloud import storage
from google.oauth2 import service_account

# ==========================================
# CONFIGURATION
# ==========================================
BUCKET_NAME = 'venice_singlepages' # Your bucket
BUCKET_PREFIX = "" # Folder prefix if applicable
CSV_FILENAME = 'detailed_page_analysis.csv' # The file to edit

st.set_page_config(layout="wide", page_title="GCS Verifier")

# ==========================================
# AUTHENTICATION (User Upload)
# ==========================================
st.sidebar.title("Login")
uploaded_key = st.sidebar.file_uploader("Upload Service Account JSON", type='json')

if not uploaded_key:
    st.warning("Please upload a Google Cloud JSON key to proceed.")
    st.stop()

# Create Credentials Object from the uploaded file
key_data = json.load(uploaded_key)
credentials = service_account.Credentials.from_service_account_info(key_data)
client = storage.Client(credentials=credentials)

# ==========================================
# GCS HELPERS
# ==========================================
def load_csv_from_gcs():
    """Tries to load the CSV from the bucket. If not found, returns None."""
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
    """Saves the DataFrame back to the bucket as a CSV."""
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        blob.upload_from_string(df.to_csv(index=False), 'text/csv')
        st.success(f"Saved {CSV_FILENAME} to Google Cloud Bucket!")
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
# APP LOGIC
# ==========================================

def init_state():
    if 'df' not in st.session_state:
        # 1. Try loading from Cloud (Current State)
        df = load_csv_from_gcs()
        
        # 2. If Cloud is empty, use the local file uploaded with the app (Initial State)
        if df is None:
            try:
                df = pd.read_csv(CSV_FILENAME)
                st.info("Loaded initial dataset from local file. Save to push to Cloud.")
            except FileNotFoundError:
                st.error("No CSV file found locally or in cloud.")
                st.stop()

        # Add tracking columns if missing
        if 'Verification_Status' not in df.columns:
            df.insert(0, 'Verification_Status', False)
        if 'Notes' not in df.columns:
            df.insert(1, 'Notes', "")
            
        st.session_state['df'] = df
        st.session_state.page_index = 0

init_state()
df = st.session_state['df']

# Navigation
unique_pages = df['Page_ID'].unique()

def next_page():
    if st.session_state.page_index < len(unique_pages) - 1:
        st.session_state.page_index += 1
def prev_page():
    if st.session_state.page_index > 0:
        st.session_state.page_index -= 1

st.sidebar.write("---")
col_prev, col_next = st.sidebar.columns(2)
with col_prev:
    st.button("Previous", on_click=prev_page)
with col_next:
    st.button("Next", on_click=next_page)

selected_page = st.sidebar.selectbox(
    "Select Image", 
    unique_pages, 
    index=st.session_state.page_index
)

if selected_page != unique_pages[st.session_state.page_index]:
    st.session_state.page_index = list(unique_pages).index(selected_page)

# Layout
col_img, col_data = st.columns([1, 1.5])

with col_img:
    st.subheader(f"Image: {selected_page}")
    blob_path = f"{BUCKET_PREFIX}/{selected_page}".replace("//", "/")
    if blob_path.startswith("/"): blob_path = blob_path[1:]
    
    with st.spinner("Loading..."):
        image_stream = load_image_from_gcs(blob_path)
    
    if image_stream:
        st.image(image_stream, use_container_width=True)
    else:
        st.error("Image not found in bucket.")

with col_data:
    st.subheader("Data Editor")
    
    page_mask = df['Page_ID'] == selected_page
    page_data = df.loc[page_mask]
    
    edited_page_data = st.data_editor(
        page_data, 
        key="editor",
        hide_index=True,
        column_config={
            "Verification_Status": st.column_config.CheckboxColumn("Verified?", default=False),
            "Notes": st.column_config.TextColumn("Notes", width="medium"),
        }
    )
    
    if not edited_page_data.equals(page_data):
        df.loc[page_mask] = edited_page_data
        st.session_state['df'] = df

    st.write("---")
    # SAVE BUTTON - Now uploads to Cloud
    if st.button("Save Changes to Cloud", type="primary"):
        save_csv_to_gcs(df)
    
    verified = df['Verification_Status'].sum()
    st.progress(verified / len(df))
    st.caption(f"Progress: {verified} / {len(df)}")