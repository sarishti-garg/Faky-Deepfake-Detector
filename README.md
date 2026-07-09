---
title: Faky Detector
emoji: 🏢
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
---

# Faky: Deepfake Image Detector

Faky is a web-based Deepfake detection application powered by **Gradio** and **PyTorch**. It utilizes an ensemble of convolutional neural network models to classify images as **Real**, **Fake**, or **Uncertain** based on confidence scores.

---

## 🚀 Features
- **Multi-Model Ensemble**: Combines predictions from multiple deep learning architectures to compute a weighted confidence score.
- **Gradio Web Interface**: A sleek, user-friendly interface allowing users to upload images and see real-time classification probability bars.
- **Inference Visualization**: Automated classification output showing probability scores.

---

## 🛠️ Model Ensemble
The detector loads and averages confidence scores from three pre-trained models:
1. **EfficientNet-B0** (`parameters/FFPP.pt`) — Weighted at **50%**
2. **MobileNet-V2** (`parameters/MobileViT_FFPP.pt`) — Weighted at **15%**
3. **MobileNet-V2 / MesoNet** (`parameters/Mesonet_FFPP.pt`) — Weighted at **35%**

A weighted average is computed:
- **Real** if Real Confidence > 0.50
- **Fake** if Fake Confidence > 0.50
- **Uncertain** if neither crosses the threshold.

---

## 📁 Project Structure

```text
├── parameters/          # Model checkpoints (.pt files) [tracked with Git LFS]
├── uploads/             # Temporary folder for uploaded images
├── .gitattributes       # Git LFS configurations
├── .gitignore           # File/folder ignore list for Git
├── app.py               # Gradio web application server
├── mesonet.py           # MesoNet (Meso4) model architecture
├── model.py             # Inference pipeline and model helper functions
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
- `FFPP.pt`
- `MobileViT_FFPP.pt`
- `Mesonet_FFPP.pt`

---

## 🏃 How to Run the App Locally

1. Start the Gradio server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```text
   http://127.0.0.1:7860/
   ```
3. Upload an image (JPG/JPEG) to check if it's real or fake.
