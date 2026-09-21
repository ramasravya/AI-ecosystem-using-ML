import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
climate = pd.read_csv("climate change dataset/climate_change_dataset.csv")

# Features
X = climate[
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
y = climate["Forest Area (%)"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

print("=" * 50)
print(" AI Ecosystem Health Prediction System ")
print("=" * 50)

# User input
temperature = float(input("Enter Average Temperature (°C): "))
co2 = float(input("Enter CO2 Emissions (Tons/Capita): "))
sea = float(input("Enter Sea Level Rise (mm): "))
rain = float(input("Enter Rainfall (mm): "))
renewable = float(input("Enter Renewable Energy (%): "))
events = int(input("Enter Extreme Weather Events: "))

# Predict
prediction = model.predict([[temperature, co2, sea, rain, renewable, events]])

forest_area = prediction[0]

print("\nPredicted Forest Area: {:.2f}%".format(forest_area))

# Ecosystem status
if forest_area >= 60:
    status = "Healthy 🌳"
elif forest_area >= 30:
    status = "Moderate ⚠️"
else:
    status = "Critical 🚨"

print("Ecosystem Status:", status)

# Sustainability recommendations
print("\nSustainability Recommendations")

if status == "Healthy 🌳":
    print("✔ Continue forest conservation.")
    print("✔ Maintain renewable energy usage.")
    print("✔ Monitor biodiversity regularly.")

elif status == "Moderate ⚠️":
    print("✔ Increase tree plantation.")
    print("✔ Reduce CO₂ emissions.")
    print("✔ Improve renewable energy usage.")

else:
    print("✔ Start reforestation programs.")
    print("✔ Reduce industrial pollution.")
    print("✔ Strengthen wildlife conservation.")
    print("✔ Promote renewable energy.")