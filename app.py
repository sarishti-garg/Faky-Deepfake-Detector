import os
import gradio as gr
from model import make_predictions

# Hugging Face ZeroGPU requirement: import spaces and decorate the prediction function
try:
    import spaces
except ImportError:
    # Dummy decorator fallback for running locally without spaces library
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# Define the models to load
model_paths = [
    "parameters/FFPP.pt",
    "parameters/MobileViT_FFPP.pt",
    "parameters/Mesonet_FFPP.pt"
]

@spaces.GPU
def predict_image(image_path):

    if not image_path:
        return "Please upload an image."
    
    # Get predictions from our model ensemble
    prediction, (real_conf, fake_conf), _ = make_predictions(image_path, model_paths)
    
    # Return dictionary with probabilities for Gradio Label output
    return {
        "Real": real_conf,
        "Fake": fake_conf
    }

# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Confidence"),
    title="Faky: Deepfake Image Detector",
    description="Upload an image to analyze and detect if it is Real or Fake using an ensemble of deep learning models.",
    examples=[
        ["uploads/deepfake2.jpg"]
    ] if os.path.exists("uploads/deepfake2.jpg") else None
)

if __name__ == '__main__':
    # Launch the Gradio app
    # Hugging Face Spaces automatically sets the port and handles routing
    demo.launch()
