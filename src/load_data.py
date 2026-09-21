import pandas as pd

# Load datasets
parks = pd.read_csv("Biodiversity/parks.csv")
species = pd.read_csv("Biodiversity/species.csv")

# Remove empty column if present
species = species.loc[:, ~species.columns.str.contains("^Unnamed")]

print("========== PARKS DATA ==========")
print(parks.head())

print("\nNumber of Parks:", parks.shape)

print("\n========== SPECIES DATA ==========")
print(species.head())

print("\nNumber of Species:", species.shape)

print("\nSpecies Data Types")
print(species.dtypes)

print("\nMissing Values")
print(species.isnull().sum())