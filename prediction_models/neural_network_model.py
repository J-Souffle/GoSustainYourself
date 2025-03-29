import pandas as pd # type: ignore
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler #type: ignroe # Changed from StandardScaler  

# Load the data (assuming tab-separated values)
input_file = "/Users/a17937/Desktop/HooHacks2025/Carbon Emission.txt"
df = pd.read_csv(input_file, delimiter="\t")

# Convert CarbonEmission to numeric (invalid parsing becomes NaN)
df["CarbonEmission"] = pd.to_numeric(df["CarbonEmission"], errors="coerce")

# Define the specified features for prediction
features = [
    "Body Type", "Sex", "Diet", "How Often Shower", "Heating Energy Source",
    "Transport", "Vehicle Type", "Social Activity", "Monthly Grocery Bill",
    "Frequency of Traveling by Air", "Vehicle Monthly Distance Km",
    "Waste Bag Size", "Waste Bag Weekly Count", "How Long TV PC Daily Hour",
    "How Many New Clothes Monthly", "How Long Internet Daily Hour",
    "Energy efficiency", "Recycling", "Cooking_With"
]

# Check that all required features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print("Missing features:", missing_features)
    exit(1)

# One-hot encode the features for the entire dataset
X_all = pd.get_dummies(df[features], drop_first=True)
y_all = df["CarbonEmission"]

# Use the first 2000 rows for training and the next 100 rows (rows 2000 to 2099) for testing
X_train = X_all.iloc[:2000]
X_test  = X_all.iloc[2000:2100]
y_train = y_all.iloc[:2000]
y_test  = y_all.iloc[2000:2100]

# Min–max normalization of features based on the training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor  = torch.FloatTensor(X_test_scaled)
y_test_tensor  = torch.FloatTensor(y_test.values).view(-1, 1)

# Create a DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define a simple neural network for regression
input_dim = X_train_tensor.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Set up loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 160
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)

# Convert PyTorch tensors to numpy arrays for metric calculation
y_true = y_test_tensor.cpu().numpy()
y_pred = predictions.cpu().numpy()

# Import metrics from scikit-learn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f"\nPrediction Metrics on Test Set:")
print(f"R^2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")