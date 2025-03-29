import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.preprocessing import StandardScaler #type: ignroe
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # type: ignore
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

# ------------------ Data Loading & Pre-processing ------------------
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
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print("Missing features:", missing_features)
    exit(1)

# One-hot encode features.
X_all = pd.get_dummies(df[features], drop_first=True)
y_all = df["CarbonEmission"]

# Scale features using StandardScaler.
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_all)

# Target normalization with StandardScaler (reversible scaling).
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_all.values.reshape(-1, 1)).flatten()

# Use the first 2100 rows; split: 2000 for training, 100 for testing.
X_sub = X_scaled[:2100]
y_sub_scaled = y_scaled[:2100]
y_sub_orig = y_all.values[:2100]  # original target values

X_train = X_sub[:2000]
X_test  = X_sub[2000:2100]
y_train_scaled = y_sub_scaled[:2000]
y_test_scaled  = y_sub_scaled[2000:2100]
# Inverting scaling for evaluation.
y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# ------------------ HistGradientBoosting Model ------------------
hgb_model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
hgb_model.fit(X_train, y_train_scaled)
y_pred_hgb_scaled = hgb_model.predict(X_test)
y_pred_hgb_orig = scaler_y.inverse_transform(y_pred_hgb_scaled.reshape(-1, 1)).flatten()

# ------------------ Evaluate Model ------------------
print("HistGradientBoosting Regression Metrics (Normalized Target):")
print(f"R²: {r2_score(y_test_scaled, y_pred_hgb_scaled):.4f}")
print(f"MAE: {mean_absolute_error(y_test_scaled, y_pred_hgb_scaled):.4f}")
print(f"MSE: {mean_squared_error(y_test_scaled, y_pred_hgb_scaled):.4f}")

print("\nHistGradientBoosting Regression Metrics (Original Target Scale):")
print(f"R²: {r2_score(y_test_orig, y_pred_hgb_orig):.4f}")
print(f"MAE: {mean_absolute_error(y_test_orig, y_pred_hgb_orig):.4f}")
print(f"MSE: {mean_squared_error(y_test_orig, y_pred_hgb_orig):.4f}")

print("\nHistGradientBoosting Per-sample Predictions (Original Target Scale):")
print("True Value | Prediction")
for true_val, pred in zip(y_test_orig, y_pred_hgb_orig):
    print(f"{true_val:10.2f} | {pred:10.2f}")

print("\nHistGradientBoosting Per-sample Predictions (Normalized Target Scale):")
print("True Value | Prediction")
for true_val, pred in zip(y_test_scaled, y_pred_hgb_scaled):
    print(f"{true_val:10.2f} | {pred:10.2f}")