import pandas as pd

# Load Climate Change Dataset
climate = pd.read_csv("climate change dataset/climate_change_dataset.csv")

# Display first 5 rows
print("========== Climate Change Dataset ==========")
print(climate.head())

# Dataset size
print("\nShape of Dataset:")
print(climate.shape)

# Column names
print("\nColumns:")
print(climate.columns)

# Data types
print("\nData Types:")
print(climate.dtypes)

# Missing values
print("\nMissing Values:")
print(climate.isnull().sum())