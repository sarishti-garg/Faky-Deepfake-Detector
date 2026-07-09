
import torch
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import timm
from mesonet import Meso4

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean & std
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
    """Loads, preprocesses, and returns an image tensor and the original image."""
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image_tensor = transform(image).unsqueeze(0)  # Apply transformations & add batch dimension
    return image, image_tensor

def combine_predictions(model1, model2, model3, inputs, device, threshold=0.5):
    """Computes weighted confidence scores for 'real' and 'fake' classification."""
    inputs = inputs.to(device)

    outputs1 = model1(inputs)
    outputs2 = model2(inputs)
    
    # MesoNet expects 128x128 input
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
