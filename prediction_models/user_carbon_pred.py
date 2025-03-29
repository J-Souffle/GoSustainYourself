from typing import Dict, Union, List
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBRegressor  # type: ignore
from lightgbm import LGBMRegressor  # type: ignore
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # type: ignore

##############################################
# ----------- HELPER FUNCTIONS -------------
##############################################
def get_valid_input(prompt: str, valid_options: List[str]) -> str:
    """
    Prompts the user until a valid option (case-insensitive) is entered.
    Returns the standardized option.
    """
    while True:
        value = input(prompt).strip()
        if value.lower() in [opt.lower() for opt in valid_options]:
            for opt in valid_options:
                if value.lower() == opt.lower():
                    return opt
        else:
            print(f"Invalid entry. Valid options are: {', '.join(valid_options)}. Please try again.")

##############################################
# ----------- DATA PREPARATION --------------
##############################################
# Load dataset.
df = pd.read_csv("prediction_models/carbon_emission_data/Carbon Emission.txt", delimiter="\t")
df["CarbonEmission"] = pd.to_numeric(df["CarbonEmission"], errors="coerce")

# Specify features.
features = [
    "Body Type", "Sex", "Diet", "How Often Shower", "Heating Energy Source",
    "Transport", "Vehicle Type", "Social Activity", "Monthly Grocery Bill",
    "Frequency of Traveling by Air", "Vehicle Monthly Distance Km",
    "Waste Bag Size", "Waste Bag Weekly Count", "How Long TV PC Daily Hour",
    "How Many New Clothes Monthly", "How Long Internet Daily Hour",
    "Energy efficiency", "Recycling", "Cooking_With"
]

# One-hot encode categorical features.
X_all = pd.get_dummies(df[features], drop_first=True)
# Clean column names if needed.
X_all.columns = X_all.columns.str.replace(r'[\[\]<>]', '_', regex=True)
y_all = df["CarbonEmission"]

# Scale input features.
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_all)

# Normalize target.
scaler_y = MinMaxScaler()
y_norm = scaler_y.fit_transform(y_all.values.reshape(-1,1)).flatten()

# Use first 2100 rows: 2000 for training, 100 for testing.
X_sub = X_scaled[:2100]
y_sub_norm = y_norm[:2100]
y_sub_orig = y_all.values[:2100]

X_train = X_sub[:2000]
X_test  = X_sub[2000:2100]
y_train_norm = y_sub_norm[:2000]
y_test_norm  = y_sub_norm[2000:2100]
y_test_orig  = scaler_y.inverse_transform(y_test_norm.reshape(-1,1)).flatten()

##############################################
# ----------- MODEL TRAINING ---------------
##############################################
# ---- Neural Network Model (PyTorch) ----
X_train_tensor = torch.FloatTensor(X_train.astype(float))
y_train_tensor = torch.FloatTensor(y_train_norm).view(-1, 1)
X_test_tensor  = torch.FloatTensor(X_test.astype(float))

input_dim = X_train_tensor.shape[1]
nn_model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
epochs = 200
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(epochs):
    nn_model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"NN Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

nn_model.eval()
with torch.no_grad():
    y_pred_nn_norm = nn_model(X_test_tensor).cpu().numpy().flatten()
y_pred_nn_orig = scaler_y.inverse_transform(y_pred_nn_norm.reshape(-1,1)).flatten()

# ---- XGBoost Model ----
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train_norm)
y_pred_xgb_norm = xgb_model.predict(X_test)
y_pred_xgb_orig = scaler_y.inverse_transform(y_pred_xgb_norm.reshape(-1,1)).flatten()

# ---- LightGBM Model ----
lgb_model = LGBMRegressor(random_state=42, n_estimators=100)
lgb_model.fit(X_train, y_train_norm)
y_pred_lgb_norm = lgb_model.predict(X_test)
y_pred_lgb_orig = scaler_y.inverse_transform(y_pred_lgb_norm.reshape(-1,1)).flatten()

# ---- Random Forest Model ----
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_norm)
y_pred_rf_norm = rf_model.predict(X_test)
y_pred_rf_orig = scaler_y.inverse_transform(y_pred_rf_norm.reshape(-1,1)).flatten()

# ---- HistGradientBoosting Model ----
hgb_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
hgb_model.fit(X_train, y_train_norm)
y_pred_hgb_norm = hgb_model.predict(X_test)
y_pred_hgb_orig = scaler_y.inverse_transform(y_pred_hgb_norm.reshape(-1,1)).flatten()

# ---- Ensemble Combination ----
combined_preds = (
    y_pred_nn_orig + y_pred_xgb_orig + y_pred_lgb_orig + y_pred_rf_orig + y_pred_hgb_orig
) / 5

##############################################
# ----------- EVALUATION OUTPUT ------------
##############################################
print("\nEnsemble Combined Model Regression Metrics (Original Target Scale):")
print(f"R²: {r2_score(y_test_orig, combined_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_test_orig, combined_preds):.4f}")
print(f"MSE: {mean_squared_error(y_test_orig, combined_preds):.4f}")

print("\nPer-sample Predictions (Original Target Scale):")
print(" True Value |   NN   |  XGB   |  LGB   |   RF   |  HGB   | Combined")
for true_val, nn_val, xgb_val, lgb_val, rf_val, hgb_val, comb_val in zip(
    y_test_orig, y_pred_nn_orig, y_pred_xgb_orig, y_pred_lgb_orig, y_pred_rf_orig, y_pred_hgb_orig, combined_preds):
    print(f"{true_val:10.2f} | {nn_val:6.2f} | {xgb_val:6.2f} | {lgb_val:6.2f} | {rf_val:6.2f} | {hgb_val:6.2f} | {comb_val:8.2f}")

##############################################
# ------- USER INTERACTION / PREDICTION ----
##############################################
print("\n-- Predict Your Carbon Emission --")

# Note: The "Body Type" entry has been removed.
valid_sexes = ["Male", "Female"]
valid_diets = ["Vegan", "Vegetarian", "Non-Vegetarian"]
valid_showers = ["Daily", "Every Other Day", "Rarely"]
valid_heating = ["Electric", "Gas", "Oil", "Other"]
valid_transport = ["Car", "Public Transport", "Bike", "Walk"]
valid_vehicle_types = ["Sedan", "SUV", "Truck", "Hatchback"]
valid_social = ["High", "Moderate", "Low"]
valid_air = ["Never", "Once a Year", "Twice a Year", "More Than Twice"]
valid_waste_sizes = ["Small", "Medium", "Large"]
valid_efficiency = ["A", "B", "C", "D", "E"]
valid_recycling = ["Always", "Sometimes", "Never"]
valid_cooking = ["Electric", "Gas", "Other"]

# Define input prompts without "Body Type".
input_prompts = {
    "Sex": lambda: get_valid_input("Enter Sex (Male, Female): ", valid_sexes),
    "Diet": lambda: get_valid_input("Enter Diet (Vegan, Vegetarian, Non-Vegetarian): ", valid_diets),
    "How Often Shower": lambda: get_valid_input("Enter How Often Shower (Daily, Every Other Day, Rarely): ", valid_showers),
    "Heating Energy Source": lambda: get_valid_input("Enter Heating Energy Source (Electric, Gas, Oil, Other): ", valid_heating),
    "Transport": lambda: get_valid_input("Enter Transport (Car, Public Transport, Bike, Walk): ", valid_transport),
    "Vehicle Type": lambda: get_valid_input("Enter Vehicle Type (Sedan, SUV, Truck, Hatchback): ", valid_vehicle_types),
    "Social Activity": lambda: get_valid_input("Enter Social Activity (High, Moderate, Low): ", valid_social),
    "Monthly Grocery Bill": lambda: input("Enter Monthly Grocery Bill (numeric): "),
    "Frequency of Traveling by Air": lambda: get_valid_input("Enter Frequency of Traveling by Air (Never, Once a Year, Twice a Year, More Than Twice): ", valid_air),
    "Vehicle Monthly Distance Km": lambda: input("Enter Vehicle Monthly Distance Km (numeric): "),
    "Waste Bag Size": lambda: get_valid_input("Enter Waste Bag Size (Small, Medium, Large): ", valid_waste_sizes),
    "Waste Bag Weekly Count": lambda: input("Enter Waste Bag Weekly Count (numeric): "),
    "How Long TV PC Daily Hour": lambda: input("Enter How Long TV PC Daily Hour (numeric): "),
    "How Many New Clothes Monthly": lambda: input("Enter How Many New Clothes Monthly (numeric): "),
    "How Long Internet Daily Hour": lambda: input("Enter How Long Internet Daily Hour (numeric): "),
    "Energy efficiency": lambda: get_valid_input("Enter Energy efficiency (A, B, C, D, E): ", valid_efficiency),
    "Recycling": lambda: get_valid_input("Enter Recycling (Always, Sometimes, Never): ", valid_recycling),
    "Cooking_With": lambda: get_valid_input("Enter Cooking_With (Electric, Gas, Other): ", valid_cooking)
}

# Dictionary to hold user inputs.
user_input: Dict[str, Union[str, float]] = {}

for feat, get_val in input_prompts.items():
    value = get_val()
    if feat in [
        "Monthly Grocery Bill", "Vehicle Monthly Distance Km",
        "Waste Bag Weekly Count", "How Long TV PC Daily Hour",
        "How Many New Clothes Monthly", "How Long Internet Daily Hour"
    ]:
        try:
            user_input[feat] = float(value)
        except ValueError:
            print(f"Invalid numeric value for {feat}. Setting to 0.")
            user_input[feat] = 0.0
    else:
        user_input[feat] = value

print("\nUser entered data:")
for k, v in user_input.items():
    print(f"{k}: {v}")

# Process user input.
user_df = pd.DataFrame([user_input])
# One-hot encode the user input.
user_encoded = pd.get_dummies(user_df, drop_first=True)
# Align user input with training feature columns.
user_encoded = user_encoded.reindex(columns=X_all.columns, fill_value=0)
# Scale the user features.
user_scaled = scaler_X.transform(user_encoded)

# Predict using each model.
user_scaled_tensor = torch.FloatTensor(user_scaled.astype(float))
with torch.no_grad():
    pred_nn_norm = nn_model(user_scaled_tensor).cpu().numpy().flatten()
pred_nn_orig = scaler_y.inverse_transform(pred_nn_norm.reshape(-1,1)).flatten()[0]

pred_xgb_norm = xgb_model.predict(user_scaled)
pred_xgb_orig = scaler_y.inverse_transform(pred_xgb_norm.reshape(-1,1)).flatten()[0]

pred_lgb_norm = lgb_model.predict(user_scaled)
pred_lgb_orig = scaler_y.inverse_transform(pred_lgb_norm.reshape(-1,1)).flatten()[0]

pred_rf_norm = rf_model.predict(user_scaled)
pred_rf_orig = scaler_y.inverse_transform(pred_rf_norm.reshape(-1,1)).flatten()[0]

pred_hgb_norm = hgb_model.predict(user_scaled)
pred_hgb_orig = scaler_y.inverse_transform(pred_hgb_norm.reshape(-1,1)).flatten()[0]

combined_pred = (pred_nn_orig + pred_xgb_orig + pred_lgb_orig + pred_rf_orig + pred_hgb_orig) / 5

print("\nPrediction for your input sample:")
print(f"Neural Network Prediction: {pred_nn_orig:.2f}")
print(f"XGBoost Prediction:        {pred_xgb_orig:.2f}")
print(f"LightGBM Prediction:       {pred_lgb_orig:.2f}")
print(f"Random Forest Prediction:  {pred_rf_orig:.2f}")
print(f"HistGradientBoosting Prediction: {pred_hgb_orig:.2f}")
print(f"Ensemble Prediction:       {combined_pred:.2f}")