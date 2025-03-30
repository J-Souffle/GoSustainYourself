import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import google.generativeai as genai
from dotenv import load_dotenv


class CarbonPredictionService:
    def __init__(self):
        # Initialize scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = None

        # Initialize models
        self.nn_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.hgb_model = None

        # Configure Gemini
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        genai.configure(api_key=api_key)
        self.model_gemini = genai.GenerativeModel('gemini-2.0-flash')

        self._load_and_train_model()

    def _load_and_train_model(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(this_dir, "../carbon_emission_data/Carbon Emission.txt")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Data file not found at: {input_file}")

        # Load and preprocess data
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

        # Validate features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Prepare features
        X_all = pd.get_dummies(df[features], drop_first=True)
        self.feature_columns = X_all.columns
        y_all = df["CarbonEmission"]

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.values.reshape(-1, 1)).flatten()

        # Initialize models
        self.nn_model = torch.nn.Sequential(
            torch.nn.Linear(X_scaled.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

        self.xgb_model = xgb.XGBRegressor(random_state=42)
        self.lgb_model = lgb.LGBMRegressor(random_state=42)
        self.rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
        self.hgb_model = HistGradientBoostingRegressor(random_state=42)

        # Train models
        X_train = X_scaled[:2000]
        y_train = y_scaled[:2000]

        # Train neural network
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.nn_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.reshape(-1, 1))
            loss.backward()
            optimizer.step()

        # Train other models
        self.xgb_model.fit(X_train, y_train)
        self.lgb_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        self.hgb_model.fit(X_train, y_train)

    def predict_carbon_emission(self, user_data: Dict[str, Any]) -> Dict[str, Union[float, str]]:
        # Prepare input
        user_df = pd.DataFrame([user_data])
        user_encoded = pd.get_dummies(user_df, drop_first=True)
        user_encoded = user_encoded.reindex(columns=self.feature_columns, fill_value=0)
        user_scaled = self.scaler_X.transform(user_encoded)
        user_scaled_tensor = torch.FloatTensor(user_scaled)

        # Get predictions
        with torch.no_grad():
            pred_nn_norm = self.nn_model(user_scaled_tensor).numpy().flatten()

        pred_xgb_norm = self.xgb_model.predict(user_scaled)
        pred_lgb_norm = self.lgb_model.predict(user_scaled)
        pred_rf_norm = self.rf_model.predict(user_scaled)
        pred_hgb_norm = self.hgb_model.predict(user_scaled)

        # Scale back predictions
        predictions = {
            "neural_network": float(self.scaler_y.inverse_transform(pred_nn_norm.reshape(-1, 1)).flatten()[0]),
            "xgboost": float(self.scaler_y.inverse_transform(pred_xgb_norm.reshape(-1, 1)).flatten()[0]),
            "lightgbm": float(self.scaler_y.inverse_transform(pred_lgb_norm.reshape(-1, 1)).flatten()[0]),
            "random_forest": float(self.scaler_y.inverse_transform(pred_rf_norm.reshape(-1, 1)).flatten()[0]),
            "hist_gradient_boosting": float(self.scaler_y.inverse_transform(pred_hgb_norm.reshape(-1, 1)).flatten()[0])
        }

        # Ensemble prediction
        combined_pred = sum(predictions.values()) / len(predictions)

        # Get suggestions
        suggestions = self._get_emission_suggestions(user_data, combined_pred)

        return {
            "prediction": combined_pred,
            "model_predictions": predictions,
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
                return response.text
            else:
                raise ValueError("Empty or invalid response from Gemini API.")
        except Exception as e:
            # Fallback suggestions in case of an error or empty response
            print(f"Error generating suggestions: {str(e)}")
            return (
                "Here are 3 general suggestions to help you reduce your carbon footprint:\n\n"
                "* **Reduce Meat Consumption:** Incorporate more plant-based meals into your diet to lower emissions from livestock farming.\n"
                "* **Optimize Energy Use:** Use energy-efficient appliances, switch to LED lighting, and unplug devices when not in use.\n"
                "* **Minimize Waste:** Recycle, compost food scraps, and reduce the use of single-use plastics."
            )


def get_valid_numeric_input(prompt: str) -> float:
    """
    Prompt the user for numeric input and validate it.
    If the input is not a valid number, remind the user and prompt again.
    """
    while True:
        user_input = input(prompt).strip()
        try:
            # Try to convert the input to a float
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_user_input() -> dict:
    """
    Collect user input dynamically for the carbon prediction chatbot.
    """
    print("Welcome to the Carbon Footprint Prediction Chatbot!")
    print("Please answer the following questions to get actionable suggestions to reduce your carbon footprint.\n")

    user_data = {
        "Body Type": input("What is your body type (e.g., Average, Slim, Overweight)? ").strip(),
        "Sex": input("What is your sex (e.g., Male, Female)? ").strip(),
        "Diet": input("What is your diet (e.g., Vegetarian, Non-Vegetarian, Vegan)? ").strip(),
        "How Often Shower": input("How often do you shower (e.g., Daily, Weekly)? ").strip(),
        "Heating Energy Source": input("What is your heating energy source (e.g., Electricity, Gas, Oil)? ").strip(),
        "Transport": input("What is your primary mode of transport (e.g., Car, Public Transport, Bicycle, Walk)? ").strip(),
    }

    # Ask for car type only if the user uses a car
    if user_data["Transport"].lower() == "car":
        user_data["Vehicle Type"] = input("If you use a car, what type is it (e.g., SUV, Sedan, Hatchback)? ").strip()

    user_data.update({
        "Social Activity": input("How would you describe your social activity level (e.g., High, Medium, Low)? ").strip(),
        "Monthly Grocery Bill": get_valid_numeric_input("What is your monthly grocery bill (e.g., 200-300)? "),
        "Frequency of Traveling by Air": input("How often do you travel by air (e.g., Rarely, Frequently)? ").strip(),
        "Vehicle Monthly Distance Km": get_valid_numeric_input("How many kilometers do you drive monthly (e.g., 500-1000)? "),
        "Waste Bag Size": input("What is the size of your waste bags (e.g., Small, Medium, Large)? ").strip(),
        "Waste Bag Weekly Count": get_valid_numeric_input("How many waste bags do you dispose of weekly (e.g., 2, 3)? "),
        "How Long TV PC Daily Hour": get_valid_numeric_input("How many hours do you spend on TV/PC daily (e.g., 4-6)? "),
        "How Many New Clothes Monthly": get_valid_numeric_input("How many new clothes do you buy monthly (e.g., 2-3)? "),
        "How Long Internet Daily Hour": get_valid_numeric_input("How many hours do you spend on the internet daily (e.g., 4-6)? "),
        "Energy efficiency": input("How would you rate your home's energy efficiency (e.g., High, Medium, Low)? ").strip(),
        "Recycling": input("How often do you recycle (e.g., Always, Sometimes, Never)? ").strip(),
        "Cooking_With": input("What is your primary cooking energy source (e.g., Electricity, Gas)? ").strip(),
    })

    return user_data


if __name__ == "__main__":
    # Collect user input
    user_data = get_user_input()

    # Initialize the CarbonPredictionService
    service = CarbonPredictionService()

    # Predict carbon emission and generate suggestions
    result = service.predict_carbon_emission(user_data)

    # Convert yearly prediction to monthly
    monthly_prediction = result['prediction']

    # Display the results
    print(f"\nPredicted Carbon Emission: {monthly_prediction:.2f} kg CO2/month")

    # Check if suggestions are valid and display them
    if result['suggestions'] and len(result['suggestions'].strip()) > 0:
        print("\nSuggestions:")
        print("Here are actionable suggestions to help you reduce your carbon footprint, focusing on areas where you can make the biggest impact:\n")
        print(result['suggestions'])
    else:
        print("\nNo actionable suggestions could be generated at this time. Please try again later.")