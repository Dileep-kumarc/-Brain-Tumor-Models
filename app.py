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
    body {
        background-color: #f7f9fc;
        font-family: "Arial", sans-serif;
    }
    .stApp {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
        border: none;
        padding: 10px 15px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Brain Tumor Detection")
st.sidebar.header("Model Status")

# -----------------------------
# 📥 DOWNLOAD MODELS
# -----------------------------
st.sidebar.subheader("Downloading Models:")
download_model_from_github("best_mri_classifier.pth", 205)
download_model_from_github("brain_tumor_classifier.h5", 134)

st.sidebar.success("✅ All models are ready!")
