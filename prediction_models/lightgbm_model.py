import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler #type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # type: ignore
from lightgbm import LGBMRegressor  # type: ignore

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

# One-hot encode features; features remain unnormalized.
X_all = pd.get_dummies(df[features], drop_first=True)
y_all = df["CarbonEmission"]

# Scale features for boosting model.
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_all)

# Normalize only the target with MinMaxScaler.
scaler_y = MinMaxScaler()
y_norm = scaler_y.fit_transform(y_all.values.reshape(-1, 1)).flatten()

# Use the first 2100 rows; split into 2000 for training and 100 for testing.
X_sub = X_scaled[:2100]
y_sub_norm = y_norm[:2100]
y_sub_orig = y_all.values[:2100]  # original target values

X_train = X_sub[:2000]
X_test  = X_sub[2000:2100]
y_train_norm = y_sub_norm[:2000]
y_test_norm  = y_sub_norm[2000:2100]
y_test_orig  = scaler_y.inverse_transform(y_test_norm.reshape(-1, 1)).flatten()

# ------------------ LightGBM Model ------------------
lgb_model = LGBMRegressor(random_state=42, n_estimators=100)
lgb_model.fit(X_train, y_train_norm)
y_pred_lgb_norm = lgb_model.predict(X_test)
y_pred_lgb_orig = scaler_y.inverse_transform(y_pred_lgb_norm.reshape(-1, 1)).flatten()

# ------------------ Evaluate Model ------------------
print("LightGBM Regression Metrics (Normalized Target):")
print(f"R²: {r2_score(y_test_norm, y_pred_lgb_norm):.4f}")
print(f"MAE: {mean_absolute_error(y_test_norm, y_pred_lgb_norm):.4f}")
print(f"MSE: {mean_squared_error(y_test_norm, y_pred_lgb_norm):.4f}")

print("\nLightGBM Regression Metrics (Original Target Scale):")
print(f"R²: {r2_score(y_test_orig, y_pred_lgb_orig):.4f}")
print(f"MAE: {mean_absolute_error(y_test_orig, y_pred_lgb_orig):.4f}")
print(f"MSE: {mean_squared_error(y_test_orig, y_pred_lgb_orig):.4f}")

print("\nLightGBM Per-sample Predictions (Original Target Scale):")
print("True Value | Prediction")
for true_val, pred in zip(y_test_orig, y_pred_lgb_orig):
    print(f"{true_val:10.2f} | {pred:10.2f}")

print("\nLightGBM Per-sample Predictions (Normalized Target Scale):")
print("True Value | Prediction")
for true_val, pred in zip(y_test_norm, y_pred_lgb_norm):
    print(f"{true_val:10.2f} | {pred:10.2f}")