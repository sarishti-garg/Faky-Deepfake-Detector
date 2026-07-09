import os
import streamlit as st
from PIL import Image
from model import make_predictions

# Page config
st.set_page_config(
    page_title="Faky: Deepfake Image Detector",
    page_icon="👁️",
    layout="centered"
)



# App Title & Description
st.title("👁️ Faky: Deepfake Image Detector")
st.write("Upload a portrait or selfie image to analyze and detect if it is **Real** or **Fake** using our ensemble of deep learning models (Xception, MobileViT, and MesoNet).")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Image uploader
uploaded_file = st.file_uploader("Choose an image (JPG or PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the file to local disk
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Run analysis
    with st.spinner("Analyzing image features..."):
        model_paths = [
            "parameters/FFPP.pt",
            "parameters/MobileViT_FFPP.pt",
            "parameters/Mesonet_FFPP.pt"
        ]
        try:
            prediction, (real_conf, fake_conf), (p1, p2, p3) = make_predictions(file_path, model_paths)
            
            
            # Display Prediction Results
            if prediction == "Real":
                st.success(f"### 🟢 Final Prediction: **{prediction}**")
            elif prediction == "Fake":
                st.error(f"### 🔴 Final Prediction: **{prediction}**")
            else:
                st.warning(f"### ⚠️ Final Prediction: **{prediction}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
