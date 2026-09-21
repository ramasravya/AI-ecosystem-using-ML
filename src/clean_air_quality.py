import pandas as pd

# Load Dataset
air = pd.read_csv(
    "air quality/AirQuality.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)

# Remove empty columns
air = air.dropna(axis=1, how="all")

# Replace -200 with missing values
air.replace(-200, pd.NA, inplace=True)

# Fill missing values
air = air.ffill()
air = air.bfill()

# Save cleaned dataset
air.to_csv("air quality/AirQuality_Cleaned.csv", index=False)

print("Missing Values After Cleaning:")
print(air.isnull().sum())

print("\nAir Quality Dataset Cleaned Successfully!")