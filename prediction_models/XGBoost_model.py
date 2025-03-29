import pandas as pd  #type: ignore
import numpy as np  #type: ignore
from sklearn.preprocessing import MinMaxScaler #type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  #type: ignore
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor  #type: ignore 

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

# One-hot encode the features; features remain unnormalized.
X_all = pd.get_dummies(df[features], drop_first=True)
# Clean column names to remove forbidden characters for XGBoost (e.g. '[', ']', '<')
X_all.columns = X_all.columns.str.replace(r'[\[\]<>]', '_', regex=True)
y_all = df["CarbonEmission"]

# ------------------ Target Normalization ------------------
# Normalize only the target (CarbonEmission).
scaler_y = MinMaxScaler()
y_all_norm = scaler_y.fit_transform(y_all.values.reshape(-1, 1)).flatten()

# Use the first 2100 rows, then split: first 2000 for training and next 100 for testing.
X_sub = X_all.iloc[:2100].copy()
y_sub_norm = y_all_norm[:2100]
y_sub_orig = y_all.iloc[:2100].values  # original target values

X_train = X_sub.iloc[:2000]
X_test  = X_sub.iloc[2000:2100]
y_train_norm = y_sub_norm[:2000]
y_test_norm  = y_sub_norm[2000:2100]

# ------------------ XGBoost Model ------------------
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train_norm)
y_pred_xgb_norm = xgb_model.predict(X_test)
# Invert normalization to obtain predictions in original scale.
y_pred_xgb_orig = scaler_y.inverse_transform(y_pred_xgb_norm.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test_norm.reshape(-1, 1)).flatten()

print("XGBoost Regression Metrics (Normalized Target):")
print(f"R^2 Score: {r2_score(y_test_norm, y_pred_xgb_norm):.4f}")
print(f"MAE: {mean_absolute_error(y_test_norm, y_pred_xgb_norm):.4f}")
print(f"MSE: {mean_squared_error(y_test_norm, y_pred_xgb_norm):.4f}")

print("\nXGBoost Regression Metrics (Original Target Scale):")
print(f"R^2 Score: {r2_score(y_test_orig, y_pred_xgb_orig):.4f}")
print(f"MAE: {mean_absolute_error(y_test_orig, y_pred_xgb_orig):.4f}")
print(f"MSE: {mean_squared_error(y_test_orig, y_pred_xgb_orig):.4f}")

print("\nXGBoost Predictions (Original Target Scale):")
for true_val, pred_val in zip(y_test_orig, y_pred_xgb_orig):
    print(f"True: {true_val:.2f}, Predicted: {pred_val:.2f}")

# ------------------ Decision Tree Model ------------------
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train_norm)
y_pred_dt_norm = dt_model.predict(X_test)
# Invert normalization for original scale
y_pred_dt_orig = scaler_y.inverse_transform(y_pred_dt_norm.reshape(-1, 1)).flatten()

print("\nDecision Tree Regression Metrics (Normalized Target):")
print(f"R^2 Score: {r2_score(y_test_norm, y_pred_dt_norm):.4f}")
print(f"MAE: {mean_absolute_error(y_test_norm, y_pred_dt_norm):.4f}")
print(f"MSE: {mean_squared_error(y_test_norm, y_pred_dt_norm):.4f}")

print("\nDecision Tree Regression Metrics (Original Target Scale):")
print(f"R^2 Score: {r2_score(y_test_orig, y_pred_dt_orig):.4f}")
print(f"MAE: {mean_absolute_error(y_test_orig, y_pred_dt_orig):.4f}")
print(f"MSE: {mean_squared_error(y_test_orig, y_pred_dt_orig):.4f}")

print("\nDecision Tree Predictions (Original Target Scale):")
for true_val, pred_val in zip(y_test_orig, y_pred_dt_orig):
    print(f"True: {true_val:.2f}, Predicted: {pred_val:.2f}")