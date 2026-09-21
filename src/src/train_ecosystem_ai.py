import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Load Climate Dataset
# ===============================

df = pd.read_csv("climate change dataset/climate_change_dataset.csv")

# ===============================
# Create Ecosystem Health Label
# ===============================

def ecosystem_status(row):

    if (
        row["Avg Temperature (°C)"] <= 28 and
        row["Rainfall (mm)"] >= 120 and
        row["Renewable Energy (%)"] >= 50
    ):
        return "Healthy"

    elif (
        row["Avg Temperature (°C)"] <= 34 and
        row["Rainfall (mm)"] >= 60
    ):
        return "Moderate"

    else:
        return "Critical"

df["Ecosystem_Health"] = df.apply(ecosystem_status, axis=1)

# ===============================
# Features
# ===============================

X = df[
    [
        "Avg Temperature (°C)",
        "CO2 Emissions (Tons/Capita)",
        "Sea Level Rise (mm)",
        "Rainfall (mm)",
        "Renewable Energy (%)",
        "Extreme Weather Events"
    ]
]

# Target

y = df["Ecosystem_Health"]

# ===============================
# Split Data
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# Train AI
# ===============================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\n=================================")
print(" AI Ecosystem Model")
print("=================================\n")

print("Accuracy :", round(accuracy*100,2), "%")

print("\nClassification Report\n")

print(classification_report(y_test,prediction))

# ===============================
# Save Model
# ===============================

joblib.dump(model,"ecosystem_ai_model.pkl")

print("\nAI Model Saved Successfully!")