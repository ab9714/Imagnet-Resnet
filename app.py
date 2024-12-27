import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import requests
from io import BytesIO

def load_model():
    # Explicitly get the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    # Get the model's preprocessing transforms
    preprocess = weights.transforms()
    return model, preprocess

# Load model and preprocessing pipeline
model, preprocess = load_model()

# Cache the labels
def get_imagenet_labels():
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(LABELS_URL)
        response.raise_for_status()
        return response.text.split("\n")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading labels: {e}")
        # Fallback to local file or raise error
        raise

LABELS = get_imagenet_labels()

def predict_image(image):
    if image is None:
        return None
    
    try:
        # Convert to RGB if image is in RGBA format
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Use the model's preprocessing pipeline
        image_tensor = preprocess(image).unsqueeze(0)
        
        # Move to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            
        # Get top 5 predictions
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Create results dictionary
        results = {LABELS[idx]: float(prob) for prob, idx in zip(top5_prob, top5_idx)}
        
        return results
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    examples=[
        ["examples/dog.jpg"],
        ["examples/cat.jpg"],
    ],
    title="ResNet50 Image Classifier",
    description="""Upload an image and the model will predict what it contains using ResNet50 trained on ImageNet.
                   The model will return the top 5 predicted classes with their confidence scores.""",
    article="""<div style='text-align: center'>
                  <p>Model: ResNet50 | Dataset: ImageNet</p>
                  <p>The model is trained on 1000 different classes from the ImageNet dataset.</p>
               </div>""",
    theme="default"
)

if __name__ == "__main__":
    iface.launch(share=False) 