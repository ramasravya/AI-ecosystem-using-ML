import pandas as pd
import matplotlib.pyplot as plt
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

# Feature Importance
importance = model.feature_importances_

# Different colors for each bar
colors = [
    "red",
    "orange",
    "blue",
    "green",
    "purple",
    "brown"
]

plt.figure(figsize=(10,6))

bars = plt.bar(
    X.columns,
    importance,
    color=colors,
    edgecolor="black"
)

# Display values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.003,
        f"{height:.2f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Feature Importance in AI Ecosystem", fontsize=16, fontweight="bold")
plt.xlabel("Environmental Factors", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)

plt.xticks(rotation=30, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()