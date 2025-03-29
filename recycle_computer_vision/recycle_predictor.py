import os
import torch
from torchvision import transforms
from PIL import Image
from training import XceptionClassifier, predict_image, GROUP_MAPPING

# Cache the loaded model so that we don't have to reload it on every call.
_model = None

def load_model(model_path, device):
    model = XceptionClassifier(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_recyclable_image(image_path):
    """
    Given an image file path, applies the necessary transforms,
    runs the model prediction, and returns a dictionary with the predicted label
    and confidence.
    """
    global _model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust model_path if necessary; here we assume the model is saved in the same directory.
    model_path = os.path.join(script_dir, "xception_garbage_classifier.pth")
    
    if _model is None:
        _model = load_model(model_path, device)
    
    # Define transform matching the training input size (299x299) and normalization.
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    pred, confidence = predict_image(_model, image_path, transform, device)
    # Reverse the GROUP_MAPPING so that we can map the predicted numeric label back to a string.
    inv_map = {v: k for k, v in GROUP_MAPPING.items()}
    label = inv_map.get(pred, "Unknown")
    return {"predicted_label": label, "confidence": confidence}