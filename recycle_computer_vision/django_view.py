import os
from tempfile import NamedTemporaryFile

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

import torch
from torchvision import transforms
from PIL import Image

# Import the Xception model and helper function from training.py
from training import XceptionClassifier, predict_image

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        uploaded_file = request.FILES['file']
        
        # Save the uploaded file to a temporary file
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Set up device and load the model from training.py
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'xception_garbage_classifier.pth')
        model = XceptionClassifier(num_classes=5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Define the transform (matching training input size 299x299)
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Run prediction using the helper function from training.py
        pred, prob = predict_image(model, tmp_path, transform, device)
        
        # Map the numeric prediction to a human-readable label
        classes = {
            0: "non-recyclable",
            1: "glass",
            2: "paper",
            3: "fabric",
            4: "recyclable-inorganic"
        }
        result_pred = classes.get(pred, "Unknown")
        
        # Clean up temporary file
        os.remove(tmp_path)
        
        return JsonResponse({"predicted_class": result_pred, "confidence": prob})
    else:
        return HttpResponse("Method Not Allowed", status=405)   