import os
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
import numpy as np
import streamlit as st

# -----------------------------
# 📥 GITHUB MODEL URLS
# -----------------------------
GITHUB_BASE_URL = "https://github.com/Dileep-kumarc/-Brain-Tumor-Models/raw/main/"

MRI_MODEL_FILENAME = "best_mri_classifier.pth"
TUMOR_MODEL_FILENAME = "brain_tumor_classifier.h5"

MRI_MODEL_PATH = os.path.join(os.getcwd(), MRI_MODEL_FILENAME)
TUMOR_MODEL_PATH = os.path.join(os.getcwd(), TUMOR_MODEL_FILENAME)

EXPECTED_FILE_SIZES = {
    MRI_MODEL_FILENAME: 205 * 1024 * 1024,  # 205 MB
    TUMOR_MODEL_FILENAME: 134 * 1024 * 1024  # 134 MB
}

# -----------------------------
# 📥 DOWNLOAD MODEL FUNCTION
# -----------------------------
def download_model(filename):
    """Download model from GitHub if not present or corrupted."""
    url = GITHUB_BASE_URL + filename
    local_path = os.path.join(os.getcwd(), filename)

    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        expected_size = EXPECTED_FILE_SIZES.get(filename, 0)

        if abs(file_size - expected_size) <= 5 * 1024 * 1024:  # Allow 5MB variance
            st.sidebar.success(f"✅ {filename} is already downloaded.")
            return local_path
        else:
            os.remove(local_path)
            st.warning(f"⚠️ Corrupt file detected for {filename}. Re-downloading...")

    # Download file
    with st.spinner(f"Downloading {filename}... ⏳"):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file size after download
            file_size = os.path.getsize(local_path)
            expected_size = EXPECTED_FILE_SIZES.get(filename, 0)

            if abs(file_size - expected_size) > 5 * 1024 * 1024:  # Allow 5MB variance
                os.remove(local_path)
                st.error(f"❌ Download failed: File size mismatch for {filename}.")
                return None

            st.sidebar.success(f"✅ {filename} downloaded successfully!")
            return local_path

        except Exception as e:
            st.error(f"❌ Failed to download {filename}: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return None

# -----------------------------
# 🎨 STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Detection")
st.sidebar.header("⚡ Model Status")

# -----------------------------
# 📥 DOWNLOAD MODELS
# -----------------------------
st.sidebar.subheader("Downloading Models:")
MRI_MODEL_PATH = download_model(MRI_MODEL_FILENAME)
TUMOR_MODEL_PATH = download_model(TUMOR_MODEL_FILENAME)

if not MRI_MODEL_PATH or not TUMOR_MODEL_PATH:
    st.error("❌ Model files missing. Please check your internet connection or model URLs.")
    st.stop()

st.sidebar.success("✅ All models are ready!")

# -----------------------------
# 🧠 LOAD MODELS
# -----------------------------
@st.cache_resource
def load_torch_model(model_path):
    """Load a PyTorch model safely."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load PyTorch model: {e}")
        return None

@st.cache_resource
def load_tf_model(model_path):
    """Load a TensorFlow model safely."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load TensorFlow model: {e}")
        return None

# Load models
mri_checker = load_torch_model(MRI_MODEL_PATH)
tumor_classifier = load_tf_model(TUMOR_MODEL_PATH)

# -----------------------------
# 🖼️ IMAGE PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image_torch(image):
    """Preprocess image for PyTorch model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

def preprocess_image_tf(image):
    """Preprocess image for TensorFlow model."""
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# -----------------------------
# 📂 FILE UPLOAD & CLASSIFICATION
# -----------------------------
st.subheader("📤 Upload an MRI Scan")
uploaded_file = st.file_uploader("Upload an MRI Image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1️⃣: Check if the uploaded image is an MRI scan
    st.subheader("🧐 Checking if this is an MRI image...")
    input_tensor = preprocess_image_torch(image)

    if mri_checker:
        with torch.no_grad():
            is_mri_output = mri_checker(input_tensor)
            is_mri = torch.argmax(is_mri_output, dim=1).item()

        if is_mri == 0:  # Assuming 0 = Not MRI, 1 = MRI
            st.error("🚫 This is NOT an MRI image. Please upload a valid brain MRI scan.")
        else:
            st.success("✅ MRI scan detected!")

            # Step 2️⃣: Classify the MRI tumor type
            st.subheader("🔬 Classifying Tumor Type...")
            input_tensor_tf = preprocess_image_tf(image)

            if tumor_classifier:
                prediction = tumor_classifier.predict(input_tensor_tf)
                predicted_class = np.argmax(prediction, axis=1)[0]

                # Mapping output to class names
                class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
                prediction_result = class_labels[predicted_class]

                # Show the classification result
                st.success(f"🧠 **Prediction:** {prediction_result}")
            else:
                st.error("❌ Tumor classification model is not loaded.")
