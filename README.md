---
title: Faky Detector
emoji: 👁️
colorFrom: gray
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
python_version: '3.12'
app_file: streamlit_app.py
pinned: false
license: mit
---

# Faky: Deepfake Image Detector

Faky is a web-based Deepfake detection application powered by **Streamlit**, **OpenCV**, and **PyTorch**. It utilizes a calibrated ensemble of deep learning convolutional neural network models to classify portrait images as **Real** or **Fake**.

---

## 🚀 Features
- **Multi-Model Ensemble**: Combines predictions from multiple deep learning architectures to compute a robust weighted average.
- **Robust Face Detection**: Automatic OpenCV Haar Cascade face crop with optimized parameters (`scaleFactor=1.05`, `minNeighbors=4`) to detect low-resolution, webcam, and tilted faces, falling back to full-image analysis when no face is present.
- **Premium Streamlit UI**: A clean, color-coded status box displaying only the final prediction (**Real** in green or **Fake** in red).

---

## 🛠️ Model Ensemble & Decision Logic
The detector loads and averages confidence scores from three pre-trained models:
1. **Xception** (`parameters/FFPP.pt`) — Weighted at **70%** (Primary robust classifier)
2. **MobileViT-XXS** (`parameters/MobileViT_FFPP.pt`) — Weighted at **15%**
3. **Meso4** (`parameters/Mesonet_FFPP.pt`) — Weighted at **15%**

### Decision Boundary
A weighted real probability score is computed. The classification is made using a calibrated fixed threshold of **`0.30`**:
- **Real** if Ensemble Real Score > 0.30
- **Fake** if Ensemble Real Score <= 0.30

This calibration prevents low-resolution camera blur and JPEG compression artifacts on real photos from triggering false "Fake" classifications, while keeping high sensitivity for actual deepfakes.

---

## 📁 Project Structure

```text
├── parameters/          # Model checkpoints (.pt files) [tracked with Git LFS]
├── uploads/             # Temporary folder for uploaded images
├── .gitattributes       # Git LFS configurations
├── .gitignore           # File/folder ignore list for Git
├── app.py               # Gradio web application (backup deployment option)
├── mesonet.py           # MesoNet (Meso4) model architecture
├── model.py             # Inference pipeline, face crop, and ensemble logic
├── streamlit_app.py     # Main Streamlit web application
└── requirements.txt     # Python dependencies
```

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd Faky
```

### 2. Set up a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
Make sure to place your model weight files inside the `parameters/` directory:
- `FFPP.pt` (Xception)
- `MobileViT_FFPP.pt` (MobileViT-XXS)
- `Mesonet_FFPP.pt` (MesoNet)

---

## 🏃 How to Run the App Locally

1. Start the Streamlit server:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your web browser and navigate to:
   ```text
   http://localhost:8501
   ```
3. Upload an image (JPG/JPEG/PNG) to analyze.
