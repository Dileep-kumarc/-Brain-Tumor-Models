import os
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import streamlit as st
from torchvision import transforms
from PIL import Image
import requests

# -----------------------------
# 📥 DOWNLOAD MODEL FILES FROM GITHUB
# -----------------------------
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Dileep-kumarc/-Brain-Tumor-Models/main/"

def download_model_from_github(filename, expected_size_mb):
    """Download model file from GitHub with size validation."""
    if not os.path.exists(filename):
        url = GITHUB_BASE_URL + filename
        st.info(f"Downloading {filename} from GitHub...")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
            if file_size < expected_size_mb * 0.8:
                st.error(f"File size mismatch for {filename}. Expected ~{expected_size_mb}MB but got {file_size:.2f}MB.")
                os.remove(filename)
                raise ValueError("Invalid file size")

            st.success(f"Successfully downloaded {filename} ({file_size:.2f} MB)")
        except Exception as e:
            st.error(f"Failed to download {filename}: {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)
            raise
    else:
        st.info(f"{filename} already exists locally.")

# Download models with expected sizes
try:
    download_model_from_github("best_mri_classifier.pth", 205)
    download_model_from_github("brain_tumor_classifier.h5", 134)
except Exception as e:
    st.error("Model download failed. Cannot proceed.")
    st.stop()

# -----------------------------
# 🧠 LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    """Load PyTorch and TensorFlow models with error handling."""
    # Define PyTorch CustomCNN
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, 2)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(-1, 128 * 28 * 28)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Load PyTorch model
    try:
        model = CustomCNN()
        state_dict = torch.load("best_mri_classifier.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        st.error(f"Failed to load PyTorch model: {str(e)}")
        raise

    # Load TensorFlow model
    try:
        classifier_model = tf.keras.models.load_model("brain_tumor_classifier.h5")
    except Exception as e:
        st.error(f"Failed to load TensorFlow model: {str(e)}")
        raise

    return model, classifier_model

# Load models with error handling
try:
    custom_cnn_model, classifier_model = load_models()
except Exception as e:
    st.error("Model loading failed. Please check logs or try again.")
    st.stop()

# -----------------------------
# 📷 IMAGE PROCESSING
# -----------------------------
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for classification."""
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:  # Convert grayscale to RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    return image_array

def validate_mri(image, model):
    """Validate if the image is an MRI using PyTorch model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    try:
        tensor_image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor_image)
            pred = torch.argmax(output, dim=1).item()
        return ("MRI", True) if pred == 0 else ("Non-MRI", False)
    except Exception as e:
        st.error(f"Error in MRI validation: {str(e)}")
        return ("Error", False)

def classify_tumor(image, model):
    """Classify tumor type using TensorFlow model."""
    try:
        image_array = preprocess_image(image)
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array, verbose=0)
        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        return classes[np.argmax(predictions)], float(np.max(predictions))
    except Exception as e:
        st.error(f"Error in tumor classification: {str(e)}")
        return "Error", 0.0

# -----------------------------
# 🎨 STREAMLIT UI WITH CSS

# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #F3F4F6; }
        .css-18e3th9 { background-color: #1E293B !important; }
        h1 { text-align: center; color: #1E40AF; }
        .css-1d391kg { background-color: #1E293B !important; color: white; }
        .stImage, .stAlert { text-align: center; }
        .stButton>button {
            background-color: #1E40AF; color: white; font-size: 16px;
            padding: 8px 20px; border-radius: 8px; border: none;
        }
        .stButton>button:hover { background-color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Detection")
st.sidebar.header("Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.header("Step 1: MRI Validation")
        image_type, is_mri = validate_mri(image, custom_cnn_model)
        if not is_mri:
            st.error(f"❌ Detected: {image_type}. Please upload a valid MRI image.")
        else:
            st.success("✅ Image validated as MRI. Proceeding to classification...")

            st.header("Step 2: Tumor Classification")
            tumor_type, confidence = classify_tumor(image, classifier_model)
            st.write(f"**Tumor Type:** `{tumor_type}` (Confidence: `{confidence:.2f}`)")

            if tumor_type == "No Tumor":
                st.info("✅ No tumor detected.")
            elif tumor_type == "Error":
                st.error("⚠ Classification failed. Please try again.")
            else:
                st.warning("⚠ Tumor detected. Consult a specialist!")
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
