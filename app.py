import os
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
import numpy as np

# -----------------------------
# 📥 GITHUB BASE URL
# -----------------------------
GITHUB_BASE_URL = "https://github.com/Dileep-kumarc/-Brain-Tumor-Models/raw/main/"

# Model filenames and expected sizes (in MB)
MODEL_FILES = {
    "best_mri_classifier.pth": 196,  # Updated size
    "brain_tumor_classifier.h5": 128  # Updated size
}

# -----------------------------
# 📥 DOWNLOAD MODEL FUNCTION
# -----------------------------
def download_model(filename, expected_size_mb):
    """Download a model file from GitHub and verify its integrity."""
    local_path = os.path.join(os.getcwd(), filename)

    # Check if the file already exists and has the correct size
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # Convert to MB
        if abs(file_size - expected_size_mb) < 5:  # Allow minor variation
            st.sidebar.success(f"✅ {filename} is ready ({file_size:.2f} MB)")
            return local_path
        else:
            st.warning(f"⚠️ Corrupt file detected for {filename}. Re-downloading...")

    # Download the file
    url = GITHUB_BASE_URL + filename
    with st.spinner(f"Downloading {filename}... ⏳"):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file size
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            if abs(file_size - expected_size_mb) > 5:
                os.remove(local_path)
                st.error(f"❌ File size mismatch for {filename}. Expected ~{expected_size_mb}MB but got {file_size:.2f}MB.")
                return None

            st.sidebar.success(f"✅ {filename} downloaded successfully ({file_size:.2f} MB)")
            return local_path

        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
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

MODEL_PATHS = {}
for filename, size in MODEL_FILES.items():
    MODEL_PATHS[filename] = download_model(filename, size)

if None in MODEL_PATHS.values():
    st.sidebar.error("❌ Model files missing. Please check your internet connection or model URLs.")
    st.stop()  # Stop execution if models are missing

# -----------------------------
# 🧠 CUSTOM CNN MODEL (PyTorch)
# -----------------------------
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)  # Output: MRI or Not MRI

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# 🧠 LOAD MODELS
# -----------------------------
@st.cache_resource
def load_torch_model(model_path):
    if not model_path or not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    try:
        model = CustomCNN()  # Ensure the correct model architecture
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading PyTorch model: {str(e)}")
        return None

@st.cache_resource
def load_tf_model(model_path):
    """Load a TensorFlow model."""
    if not model_path or not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Error loading TensorFlow model: {str(e)}")
        return None

# Load models
mri_checker = load_torch_model(MODEL_PATHS["best_mri_classifier.pth"])
tumor_classifier = load_tf_model(MODEL_PATHS["brain_tumor_classifier.h5"])

# -----------------------------
# 🖼️ IMAGE PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image_torch(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)

def preprocess_image_tf(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

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
            st.stop()
        else:
            st.success("✅ MRI scan detected!")

            # Step 2️⃣: Classify the MRI tumor type
            st.subheader("🔬 Classifying Tumor Type...")
            input_tensor_tf = preprocess_image_tf(image)

            if tumor_classifier:
                prediction = tumor_classifier.predict(input_tensor_tf)
                predicted_class = np.argmax(prediction, axis=1)[0]
                class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
                prediction_result = class_labels[predicted_class]
                st.success(f"🧠 **Prediction:** {prediction_result}")
            else:
                st.error("❌ Tumor classification model is not loaded.")
