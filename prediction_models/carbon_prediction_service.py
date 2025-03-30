import os
import pandas as pd #type: ignore
import numpy as np
from typing import Dict, Union, Any
from sklearn.preprocessing import StandardScalern #type: ignore
from sklearn.ensemble import HistGradientBoostingRegressor #type: ignore
import google.generativeai as genai
from django.conf import settings #type: ignore


class CarbonPredictionService:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = None
        self._load_and_train_model()

        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_gemini = genai.GenerativeModel('gemini-2.0-flash')

    def _load_and_train_model(self):
        # Load dataset
        input_file = "prediction_models/carbon_emission_data/Carbon Emission.txt"
        df = pd.read_csv(input_file, delimiter="\t")
        df["CarbonEmission"] = pd.to_numeric(df["CarbonEmission"], errors="coerce")

        features = [
            "Body Type", "Sex", "Diet", "How Often Shower", "Heating Energy Source",
            "Transport", "Vehicle Type", "Social Activity", "Monthly Grocery Bill",
            "Frequency of Traveling by Air", "Vehicle Monthly Distance Km",
            "Waste Bag Size", "Waste Bag Weekly Count", "How Long TV PC Daily Hour",
            "How Many New Clothes Monthly", "How Long Internet Daily Hour",
            "Energy efficiency", "Recycling", "Cooking_With"
        ]

        # One-hot encode features
        X_all = pd.get_dummies(df[features], drop_first=True)
        self.feature_columns = X_all.columns
        y_all = df["CarbonEmission"]

        # Scale features
        X_scaled = self.scaler_X.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.values.reshape(-1, 1)).flatten()

        # Train model
        self.model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        self.model.fit(X_scaled[:2000], y_scaled[:2000])  # Using first 2000 samples for training

    def predict_carbon_emission(self, user_data: Dict[str, Any]) -> Dict[str, Union[float, str]]:
        # Create DataFrame from user input
        user_df = pd.DataFrame([user_data])

        # One-hot encode user input
        user_encoded = pd.get_dummies(user_df, drop_first=True)

        # Align user input with training features
        user_encoded = user_encoded.reindex(columns=self.feature_columns, fill_value=0)

        # Scale features
        user_scaled = self.scaler_X.transform(user_encoded)

        # Make prediction
        pred_scaled = self.model.predict(user_scaled)
        prediction = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

        # Get suggestions from Gemini
        suggestions = self._get_emission_suggestions(user_data, prediction)

        return {
            "prediction": float(prediction),
            "suggestions": suggestions
        }

    def _get_emission_suggestions(self, user_data: Dict[str, Any], prediction: float) -> str:
        try:
            prompt = (
                f"Based on the following user data and predicted carbon emission of {prediction:.2f} kg CO2/year:\n\n"
                f"{user_data}\n\n"
                "Please provide 3-5 specific, actionable suggestions to help reduce your carbon footprint. "
                "Focus on the areas where you can make the biggest impact based on your current habits. "
                "Format the response in bullet points and keep suggestions practical and achievable."
            )

            response = self.model_gemini.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )

            # Check if the response contains valid suggestions
            if response.text and len(response.text.strip()) > 0:
                suggestions = response.text.strip()
                # Format the suggestions into an HTML unordered list
                formatted_suggestions = "<ul>" + "".join(
                    [f"<li>{line.strip()}</li>" for line in suggestions.split("\n") if line.strip()]
                ) + "</ul>"
                return formatted_suggestions
            else:
                raise ValueError("Empty or invalid response from Gemini API.")
        except Exception as e:
            # Fallback suggestions in case of an error or empty response
            print(f"Error generating suggestions: {str(e)}")
            return (
                "<p>Unable to generate suggestions at this time. Here are some general tips:</p>"
                "<ul>"
                "<li><strong>Reduce Meat Consumption:</strong> Incorporate more plant-based meals into your diet to lower emissions from livestock farming.</li>"
                "<li><strong>Optimize Energy Use:</strong> Use energy-efficient appliances, switch to LED lighting, and unplug devices when not in use.</li>"
                "<li><strong>Minimize Waste:</strong> Recycle, compost food scraps, and reduce the use of single-use plastics.</li>"
                "</ul>"
            )