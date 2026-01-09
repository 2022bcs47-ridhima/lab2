# Load dataset
import json
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

# Pre-processing & Feature Selection

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Model

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation Metrics

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save Model

os.makedirs("output/model", exist_ok=True)
joblib.dump(model, "output/model/trained_model.pkl")

# Save Metrics to JSON

os.makedirs("output/results", exist_ok=True)

metrics = {
    "MSE": mse,
    "R2": r2
}

with open("output/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Name: Ridhima Edimadakala, Roll No: 2022BCS0047")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
