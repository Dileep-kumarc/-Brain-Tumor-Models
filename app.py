import os
import requests
import streamlit as st
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from PIL import Image
from torchvision import transforms

# -----------------------------
# 📥 GITHUB BASE URL
# -----------------------------
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Dileep-kumarc/Brain-Tumor-Models/main/"

# -----------------------------
# 📥 DOWNLOAD MODEL FUNCTION
# -----------------------------
def download_model_from_github(filename, expected_size_mb):
    """Download a model file from GitHub and check its integrity."""
    url = GITHUB_BASE_URL + filename
    with st.spinner(f"Downloading {filename}... ⏳"):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
            if file_size < expected_size_mb * 0.8:
                os.remove(filename)
                st.error(f"❌ File size mismatch for {filename}. Expected ~{expected_size_mb}MB but got {file_size:.2f}MB.")
                return False
            st.success(f"✅ {filename} downloaded successfully ({file_size:.2f} MB)")
            return True
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Failed to download {filename}: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Unexpected error downloading {filename}: {str(e)}")
            return False

def validate_and_load_file(filename, expected_size_mb):
    """Validate file existence and integrity, attempt download if missing or invalid."""
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / (1024 * 1024)
        st.info(f"{filename} exists locally ({file_size:.2f} MB).")
        if file_size < expected_size_mb * 0.5:
            st.warning(f"{filename} size too small. Attempting re-download...")
            os.remove(filename)
            return download_model_from_github(filename, expected_size_mb)
        return True
    else:
        return download_model_from_github(filename, expected_size_mb)

# -----------------------------
# 📥 DOWNLOAD MODELS
# -----------------------------
MRI_CHECKER_MODEL_PATH = "best_mri_classifier.pth"
TUMOR_CLASSIFIER_MODEL_PATH = "brain_tumor_classifier.h5"
MRI_EXPECTED_SIZE_MB = 205  # Based on prior context
TUMOR_EXPECTED_SIZE_MB = 134  # Adjusted from prior context

mri_model_ready = validate_and_load_file(MRI_CHECKER_MODEL_PATH, MRI_EXPECTED_SIZE_MB)
tumor_model_ready = validate_and_load_file(TUMOR_CLASSIFIER_MODEL_PATH, TUMOR_EXPECTED_SIZE_MB)

if mri_model_ready and tumor_model_ready:
    st.sidebar.success("✅ All models are ready!")
else:
    st.sidebar.error("❌ Model setup failed. Check GitHub or upload models manually.")

# -----------------------------
# 🧠 LOAD MODELS
# -----------------------------
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)  # Binary: MRI or not

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_pytorch_model(model_path):
    """Load PyTorch model with fallback for state dict."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    try:
        loaded_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model = CustomCNN()
        if isinstance(loaded_data, dict):
            model.load_state_dict(loaded_data)
        else:
            model = loaded_data
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load PyTorch model: {str(e)}")
        return None

@st.cache_resource
def load_tf_model(model_path):
    """Load TensorFlow model."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load TensorFlow model: {str(e)}")
        return None

mri_checker = load_pytorch_model(MRI_CHECKER_MODEL_PATH)
tumor_classifier = load_tf_model(TUMOR_CLASSIFIER_MODEL_PATH)

# -----------------------------
# 🖼️ IMAGE PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image_pytorch(image):
    """Preprocess image for PyTorch model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def preprocess_image_tf(image):
    """Preprocess image for TensorFlow model."""
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# -----------------------------
# 🎨 STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection")
st.sidebar.header("⚡ Model Status")

# -----------------------------
# 📂 FILE UPLOAD & CLASSIFICATION
# -----------------------------
st.subheader("📤 Upload an MRI Scan")
uploaded_file = st.file_uploader("Upload an MRI Image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1️⃣: Check if the image is an MRI scan
    st.subheader("🧐 Checking if this is an MRI image...")
    if mri_checker:
        input_tensor = preprocess_image_pytorch(image)
        with torch.no_grad():
            is_mri_output = mri_checker(input_tensor)
            is_mri = torch.argmax(is_mri_output, dim=1).item()
        
        if is_mri == 0:  # Assuming 0 = Not MRI, 1 = MRI
            st.error("🚫 This is **not** an MRI image. Please upload a valid brain MRI scan.")
        else:
            st.success("✅ MRI scan detected!")

            # Step 2️⃣: Classify the MRI tumor type
            st.subheader("🔬 Classifying Tumor Type...")
            if tumor_classifier:
                input_tensor_tf = preprocess_image_tf(image)
                prediction = tumor_classifier.predict(input_tensor_tf)
                predicted_class = np.argmax(prediction, axis=1)[0]
                class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
                prediction_result = class_labels[predicted_class]
                confidence = prediction[0][predicted_class]

                st.success(f"🧠 **Prediction:** {prediction_result} (Confidence: {confidence:.2f})")
            else:
                st.error("❌ Tumor classification model is not loaded.")
    else:
        st.error("❌ MRI checker model is not loaded.")
