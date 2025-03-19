import os
import requests
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf  # For loading brain_tumor_classifier.h5

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
# 📥 DOWNLOAD MODELS
# -----------------------------
MRI_CHECKER_MODEL_PATH = "best_mri_classifier.pth"
TUMOR_CLASSIFIER_MODEL_PATH = "brain_tumor_classifier.h5"

mri_model_ready = download_model_from_github(MRI_CHECKER_MODEL_PATH, 50)  # Replace with actual size
tumor_model_ready = download_model_from_github(TUMOR_CLASSIFIER_MODEL_PATH, 205)

if mri_model_ready and tumor_model_ready:
    st.sidebar.success("✅ All models are ready!")
else:
    st.sidebar.error("❌ Model download failed. Please check the GitHub repository.")

# -----------------------------
# 🧠 LOAD MODELS
# -----------------------------
@st.cache_resource
def load_pytorch_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

@st.cache_resource
def load_tf_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found.")
        return None
    return tf.keras.models.load_model(model_path)

# Load models
mri_checker = load_pytorch_model(MRI_CHECKER_MODEL_PATH)
tumor_classifier = load_tf_model(TUMOR_CLASSIFIER_MODEL_PATH)

# -----------------------------
# 🖼️ IMAGE PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image_pytorch(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def preprocess_image_tf(image):
    image = image.resize((224, 224))  # Resize for TensorFlow model
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
    return tf.expand_dims(image, axis=0)  # Add batch dimension

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
    input_tensor = preprocess_image_pytorch(image)

    if mri_checker:
        with torch.no_grad():
            is_mri_output = mri_checker(input_tensor)
            is_mri = torch.argmax(is_mri_output, dim=1).item()

        if is_mri == 0:  # Assuming 0 = Not MRI, 1 = MRI
            st.error("🚫 This is **not** an MRI image. Please upload a valid brain MRI scan.")
        else:
            st.success("✅ MRI scan detected!")

            # Step 2️⃣: Classify the MRI tumor type
            st.subheader("🔬 Classifying Tumor Type...")
            input_tensor_tf = preprocess_image_tf(image)

            if tumor_classifier:
                prediction = tumor_classifier.predict(input_tensor_tf)
                predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

                # Mapping output to class names
                class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
                prediction_result = class_labels[predicted_class]

                # Show the classification result
                st.success(f"🧠 **Prediction:** {prediction_result}")
            else:
                st.error("❌ Tumor classification model is not loaded.")
