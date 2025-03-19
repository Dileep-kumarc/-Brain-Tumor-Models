import os
import requests
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

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
                if response.status_code == 404:
                    st.error(f"❌ File not found: {filename}. Please check if it's uploaded to GitHub.")
                    return False
                
                response.raise_for_status()

                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Verify file size
                file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
                if file_size < expected_size_mb * 0.8:
                    os.remove(filename)
                    st.error(f"❌ File size mismatch for {filename}. Expected ~{expected_size_mb}MB but got {file_size:.2f}MB.")
                    return False

                st.success(f"✅ {filename} downloaded successfully ({file_size:.2f} MB)")
                return True

            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                if os.path.exists(filename):
                    os.remove(filename)
                return False
    return True

# -----------------------------
# 🎨 STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Detection")
st.sidebar.header("⚡ Model Status")

# -----------------------------
# 📥 DOWNLOAD MODEL
# -----------------------------
TUMOR_CLASSIFIER_MODEL_PATH = "best_mri_classifier.pth"

if download_model_from_github(TUMOR_CLASSIFIER_MODEL_PATH, 205):
    st.sidebar.success("✅ Model is ready!")
else:
    st.sidebar.error("❌ Model download failed. Please check the GitHub repository.")

# -----------------------------
# 🧠 LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found. Please check your download URL.")
        return None
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Load the tumor classification model
tumor_classifier = load_model(TUMOR_CLASSIFIER_MODEL_PATH)

# -----------------------------
# 🖼️ IMAGE PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to range [-1,1]
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# -----------------------------
# 📂 FILE UPLOAD & CLASSIFICATION
# -----------------------------
st.subheader("📤 Upload an MRI Scan")
uploaded_file = st.file_uploader("Upload an MRI Image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1️⃣: Classify the MRI tumor type
    st.subheader("🔬 Classifying Tumor Type...")
    input_tensor = preprocess_image(image)

    if tumor_classifier:
        with torch.no_grad():
            tumor_output = tumor_classifier(input_tensor)
            predicted_class = torch.argmax(tumor_output, dim=1).item()

        # Mapping output to class names
        class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
        prediction_result = class_labels[predicted_class]

        # Show the classification result
        st.success(f"🧠 **Prediction:** {prediction_result}")
    else:
        st.error("❌ Tumor classification model is not loaded.")
