import pandas as pd

# Load Forest Fire Dataset
forest = pd.read_csv("forestfire/forestfires.csv")

# Display first 5 rows
print("========== Forest Fire Dataset ==========")
print(forest.head())

# Dataset size
print("\nShape of Dataset:")
print(forest.shape)

# Column names
print("\nColumns:")
print(forest.columns)

# Data types
print("\nData Types:")
print(forest.dtypes)

# Missing values
print("\nMissing Values:")
print(forest.isnull().sum())