import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_gemini_explanation(prediction_result, image_path=None):
    load_dotenv()
    """
    Generate a detailed explanation for the prediction result by querying the Gemini API.
    
    Args:
        prediction_result (dict): Contains 'predicted_label' and 'confidence'.
        
    Returns:
        str: Detailed explanation from the Gemini model.
    """
    label = prediction_result.get("predicted_label", "Unknown")
    confidence = prediction_result.get("confidence", 0)

    # Use hardcoded API key directly as a fallback
    api_key = "AIzaSyA6rPOBgOhvIKg1rduUVRCUzOoGUXUUBKs"
    
    # Build the Gemini API url with the API key
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    # Prepare the instruction text based on the prediction.
    instruction_text = (
        f"The image was identified as '{label}' with {confidence:.0%} confidence. "
        f"Give a BRIEF, simple explanation (2-3 sentences max) about this material as if explaining to someone with little knowledge. "
        f"Then provide a clear, practical recommendation on how they should dispose of this material to benefit the environment. "
        f"Use simple language and focus on actionable advice."
    )

    # Build the payload as required by the Gemini API.
    payload = {
        "contents": [{
            "parts": [{"text": instruction_text}]
        }]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(gemini_api_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract the explanation from the response.
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        
        # Fallback explanation with material-specific recommendations
        return get_material_specific_fallback(label, confidence)
    except Exception as e:
        print(f"API Error: {str(e)}")
        return get_material_specific_fallback(label, confidence)

def get_material_specific_fallback(label, confidence):
    """Provides material-specific recommendations when Gemini API fails."""
    base_message = f"Material identified: {label} (confidence: {confidence:.0%}). "
    
    if "glass" in label.lower():
        return base_message + "Glass is infinitely recyclable without quality loss. Place in glass recycling bins, ensuring items are clean and sorted by color when required by local guidelines."
    elif "paper" in label.lower():
        return base_message + "Paper is highly recyclable. Place clean, dry paper in designated paper recycling. Avoid contamination with food or liquids to maintain recyclability."
    elif "metal" in label.lower():
        return base_message + "Metal is a valuable recyclable material. Clean and place in metal recycling. Note that some facilities separate aluminum from other metals."
    elif "plastic" in label.lower():
        return base_message + "Check the plastic recycling number (1-7) on the item. Follow local guidelines as plastic recycling varies by region and type."
    elif "cardboard" in label.lower():
        return base_message + "Flatten cardboard to save space. Ensure it's clean and dry, then place in paper/cardboard recycling. Remove any tape or non-paper materials."
    elif "organic" in label.lower() or "food" in label.lower():
        return base_message + "Compost if possible. Organic waste in landfills produces methane. Many communities offer composting services or guidelines for home composting."
    else:
        return base_message + "Follow local waste management guidelines for proper disposal. When in doubt, check with your local recycling center for specific instructions."

def test_gemini_explanation():
    """
    Test the Gemini explanation function with a sample prediction result.
    """
    sample_prediction = {
        "predicted_label": "glass",
        "confidence": 0.95
    }
    
    explanation = get_gemini_explanation(sample_prediction)
    print("Gemini Explanation:")
    print(explanation)

if __name__ == "__main__":
    test_gemini_explanation()