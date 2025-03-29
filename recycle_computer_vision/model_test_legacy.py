import os
import random
import torch
from torchvision import transforms

# Import functions from training.py (do not import load_model since we define it here)
from training import SimpleCNN, predict_image, diagnose_recyclability

def load_model(model_path, device):
    """
    Instantiate the SimpleCNN model, load saved weights and set evaluation mode.
    """
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def parse_jpg(image_path):
    """
    Uses the parent folder name as the object name and returns (image_path, label)
    """
    parent_folder = os.path.basename(os.path.dirname(image_path))
    label = diagnose_recyclability(parent_folder)
    return (image_path, label)

def run_classification_on_all_data():
    # Get the directory where this script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Path where the model weights are saved.
    model_path = os.path.join(script_dir, 'recyclable_classifier.pth')
    model = load_model(model_path, device)

    # Define the same transform used during training/validation.
    img_height, img_width = 224, 224
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Base directory where your image files are located.
    base_dir = os.path.join(script_dir, 'data', 'Main Dataset')
    print("Predictions from standalone image files:")

    # Collect image files.
    image_paths = []
    for subdir, dirs, files in os.walk(base_dir):
        for file in files:
            lower = file.lower()
            full_path = os.path.join(subdir, file)
            if lower.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(full_path)

    if image_paths:
        sample_count = min(50, len(image_paths))
        random_image_samples = random.sample(image_paths, sample_count)
        print("\n--- Stand-alone image samples ---")
        for image_path in random_image_samples:
            image_path, actual = parse_jpg(image_path)
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue
            pred, prob = predict_image(model, image_path, transform, device)
            result_pred = "Recyclable" if pred == 1 else "Non-Recyclable"
            result_actual = "Recyclable" if actual == 1 else "Non-Recyclable"
            print(f"Image: {image_path}")
            print(f"    Predicted: {result_pred} (confidence: {prob:.4f})")
            print(f"    Actual:    {result_actual}")
    else:
        print("No standalone image files found.")

if __name__ == '__main__':
    run_classification_on_all_data()