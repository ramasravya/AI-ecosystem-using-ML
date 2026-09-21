import pandas as pd

# Load Air Quality Dataset
air = pd.read_csv(
    "air quality/AirQuality.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)

# Remove empty columns
air = air.loc[:, ~air.columns.str.contains("^Unnamed")]
air = air.dropna(axis=1, how="all")

# Display first 5 rows
print("========== Air Quality Dataset ==========")
print(air.head())

# Dataset size
print("\nShape of Dataset:")
print(air.shape)

# Column names
print("\nColumns:")
print(air.columns)

# Data types
print("\nData Types:")
print(air.dtypes)

# Missing values
print("\nMissing Values:")
print(air.isnull().sum())