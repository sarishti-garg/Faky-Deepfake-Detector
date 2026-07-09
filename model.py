
import torch
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import timm
from mesonet import Meso4
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

# The base transform only converts the image to a tensor in [0, 1] range.
# Sizing and normalization are handled dynamically for each model in combine_predictions.
transform = transforms.Compose([
    transforms.ToTensor()
])

def load_models(model_paths):
    """Load pretrained models and their trained weights."""

    model1_path = model_paths[0]
    model2_path = model_paths[1]
    model3_path = model_paths[2]

    model1 = timm.create_model('xception', pretrained=False, num_classes=1)
    model2 = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=1)
    model3 = Meso4(num_classes=1)

    checkpoint1 = torch.load(model1_path, map_location=torch.device('cpu'), weights_only=False)
    checkpoint2 = torch.load(model2_path, map_location=torch.device('cpu'), weights_only=False)
    checkpoint3 = torch.load(model3_path, map_location=torch.device('cpu'), weights_only=False)

    model1.load_state_dict(checkpoint1.get("model_state_dict", checkpoint1), strict=True)
    model2.load_state_dict(checkpoint2.get("model_state_dict", checkpoint2), strict=True)
    model3.load_state_dict(checkpoint3.get("model_state_dict", checkpoint3), strict=True)

    model1.eval()
    model2.eval()
    model3.eval()

    return model1, model2, model3

def load_image(image_path):
    """Loads, detects/crops faces, preprocesses, and returns the image and tensor."""
    # Try to detect and crop face using OpenCV Haar Cascade
    img = cv2.imread(image_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # Crop the first detected face with a 15% margin
            x, y, w, h = faces[0]
            margin = int(0.15 * w)
            h_img, w_img, _ = img.shape
            y1 = max(0, y - margin)
            y2 = min(h_img, y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(w_img, x + w + margin)
            face = img[y1:y2, x1:x2]
            image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        else:
            image = Image.open(image_path).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def combine_predictions(model1, model2, model3, inputs, device, threshold=0.5):
    """Computes weighted confidence scores for 'real' and 'fake' classification."""
    # inputs is a [0, 1] range tensor
    inputs = inputs.to(device)

    # Model 1 (Xception) expects 224x224 and ImageNet normalization
    inputs_m1 = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    inputs_m1 = (inputs_m1 - mean) / std
    outputs1 = model1(inputs_m1)

    # Model 2 (MobileViT) expects 256x256 simple [0, 1] scaling
    inputs_m2 = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)
    outputs2 = model2(inputs_m2)
    
    # Model 3 (MesoNet) expects 128x128 simple [0, 1] scaling
    inputs_m3 = F.interpolate(inputs, size=(128, 128), mode='bilinear', align_corners=False)
    outputs3 = model3(inputs_m3)

    probs1 = torch.sigmoid(outputs1)
    probs2 = torch.sigmoid(outputs2)
    probs3 = outputs3  # Meso4 output is already sigmoid-activated

    weight1, weight2, weight3 = 0.50, 0.15, 0.35

    weighted_avg_fake_confidence = (weight1 * (1 - probs1) + weight2 * (1 - probs2) + weight3 * (1 - probs3))
    weighted_avg_real_confidence = (weight1 * probs1 + weight2 * probs2 + weight3 * probs3)

    if weighted_avg_real_confidence > threshold:
        final_prediction = "Real"
    elif weighted_avg_fake_confidence > threshold:
        final_prediction = "Fake"
    else:
        final_prediction = "Uncertain"

    return final_prediction, weighted_avg_fake_confidence.item(), weighted_avg_real_confidence.item()

def make_predictions(image_path, model_paths):
    """Loads models, processes the image, and returns classification result with visualization."""

    model1, model2, model3 = load_models(model_paths)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model2.to(device)
    model3.to(device)

    image, inputs = load_image(image_path)
    inputs = inputs.to(device)

    prediction, fake_conf, real_conf = combine_predictions(model1, model2, model3, inputs, device)

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Prediction: {prediction}\nReal: {real_conf:.4f}, Fake: {fake_conf:.4f}", fontsize=12, color='black')
        plt.show()
    except Exception:
        pass


    print(f"Image: {image_path}")
    print(f"Real Confidence: {real_conf:.4f}, Fake Confidence: {fake_conf:.4f}")
    print(f"Final Prediction: {prediction}")

    return prediction, (real_conf, fake_conf)
