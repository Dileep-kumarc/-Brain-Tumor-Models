import os
import requests
import streamlit as st

# -----------------------------
# 📥 GITHUB BASE URL
# -----------------------------
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Dileep-kumarc/Brain-Tumor-Models/main/"

# -----------------------------
# 📥 DOWNLOAD MODEL FUNCTION
# -----------------------------
def download_model_from_github(filename, expected_size_mb):
    """Download a model file from GitHub and check its integrity."""
    if not os.path.exists(filename):
        url = GITHUB_BASE_URL + filename
        with st.spinner(f"Downloading {filename}... ⏳"):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Verify file size
                file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
                if file_size < expected_size_mb * 0.8:
                    os.remove(filename)
                    st.error(f"File size mismatch for {filename}. Expected ~{expected_size_mb}MB but got {file_size:.2f}MB.")
                    raise ValueError("Invalid file size")

                st.success(f"✅ {filename} downloaded successfully ({file_size:.2f} MB)")

            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                if os.path.exists(filename):
                    os.remove(filename)
                raise

# -----------------------------
# 🎨 STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    /* ------------------ BODY STYLING ------------------ */
    body {
        background-color: #f4f7f6;
        font-family: "Poppins", sans-serif;
    }

    /* ------------------ APP MAIN CONTAINER ------------------ */
    .stApp {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 3px 3px 15px rgba(0,0,0,0.1);
    }

    /* ------------------ SIDEBAR STYLING ------------------ */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        padding: 20px;
        border-radius: 0 12px 12px 0;
    }
    [data-testid="stSidebar"] h2 {
        color: #fff;
        text-align: center;
    }

    /* ------------------ BUTTON STYLING ------------------ */
    .stButton>button {
        color: white;
        background: #4CAF50;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background: #45a049;
        transform: scale(1.05);
    }

    /* ------------------ TEXT INPUT STYLING ------------------ */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        padding: 8px;
    }

    /* ------------------ FILE UPLOAD BOX ------------------ */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 12px;
        padding: 10px;
        background: rgba(76, 175, 80, 0.1);
        text-align: center;
    }

    /* ------------------ SUCCESS MESSAGE ------------------ */
    .stAlert[data-baseweb="notification"][aria-live="assertive"] {
        background: #dff0d8;
        color: #3c763d;
        border-radius: 10px;
        padding: 12px;
        border-left: 5px solid #4CAF50;
    }

    /* ------------------ ERROR MESSAGE ------------------ */
    .stAlert[data-baseweb="notification"][aria-live="polite"] {
        background: #f2dede;
        color: #a94442;
        border-radius: 10px;
        padding: 12px;
        border-left: 5px solid #a94442;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 🏠 MAIN UI
# -----------------------------
st.title("🧠 Brain Tumor Detection")
st.sidebar.header("⚡ Model Status")

# -----------------------------
# 📥 DOWNLOAD MODELS
# -----------------------------
st.sidebar.subheader("📦 Downloading Models:")
download_model_from_github("best_mri_classifier.pth", 205)
download_model_from_github("brain_tumor_classifier.h5", 134)

st.sidebar.success("✅ All models are ready!")

# -----------------------------
# 📂 FILE UPLOAD AREA
# -----------------------------
st.subheader("📤 Upload an MRI Scan")
uploaded_file = st.file_uploader("Upload an MRI Image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    st.success("✅ Image Uploaded Successfully!")
