import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Climate Change Dataset
climate = pd.read_csv("climate change dataset/climate_change_dataset.csv")

# Features (Input)
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

# Target (Output)
y = climate["Forest Area (%)"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("========== AI Ecosystem Health Model ==========")
print("Model trained successfully!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))