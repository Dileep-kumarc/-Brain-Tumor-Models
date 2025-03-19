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

# ---------------
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Detection")
st.sidebar.header("⚡ Model Status")

# -----------------------------
# 📥 DOWNLOAD MODELS
# -----------------------------
MRI_CHECKER_MODEL_PATH = "mri_checker.pth"
TUMOR_CLASSIFIER_MODEL_PATH = "best_mri_classifier.pth"

download_model_from_github(MRI_CHECKER_MODEL_PATH, 50)   # Replace 50 with actual size of mri_checker.pth
download_model_from_github(TUMOR_CLASSIFIER_MODEL_PATH, 205)

st.sidebar.success("✅ All models are ready!")

# -----------------------------
# 🧠 LOAD MODELS
# -----------------------------
@st.cache_resource
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load on CPU
    model.eval()  # Set to evaluation mode
    return model

# Load both models
mri_checker = load_model(MRI_CHECKER_MODEL_PATH)
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
    
    # Step 1️⃣: Check if the uploaded image is an MRI scan
    st.subheader("🧐 Checking if this is an MRI image...")
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        is_mri_output = mri_checker(input_tensor)
        is_mri = torch.argmax(is_mri_output, dim=1).item()

    if is_mri == 0:  # Assuming 0 = Not MRI, 1 = MRI
        st.error("🚫 This is **not** an MRI image. Please upload a valid brain MRI scan.")
    else:
        st.success("✅ MRI scan detected!")

        # Step 2️⃣: Classify the MRI tumor type
        st.subheader("🔬 Classifying Tumor Type...")
        with torch.no_grad():
            tumor_output = tumor_classifier(input_tensor)
            predicted_class = torch.argmax(tumor_output, dim=1).item()
        
        # Mapping output to class names
        class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
        prediction_result = class_labels[predicted_class]
        
        # Show the classification result
        st.success(f"🧠 **Prediction:** {prediction_result}")
