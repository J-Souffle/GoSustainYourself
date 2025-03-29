import pandas as pd
import numpy as np
from typing import Dict, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import google.generativeai as genai
from django.conf import settings

class CarbonPredictionService:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = None
        self._load_and_train_model()
        
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_gemini = genai.GenerativeModel('gemini-pro')
        
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
        prompt = f"""
        Based on the following user data and their predicted carbon emission of {prediction:.2f}:
        
        {user_data}
        
        Please provide 3-5 specific, actionable suggestions to help reduce their carbon footprint. 
        Focus on the areas where they could make the biggest impact based on their current habits.
        Format the response in bullet points.
        """
        
        try:
            response = self.model_gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Unable to generate suggestions at this time. Error: {str(e)}"