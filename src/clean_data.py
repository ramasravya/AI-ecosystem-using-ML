import pandas as pd

# Load datasets
parks = pd.read_csv("Biodiversity/parks.csv")
species = pd.read_csv("Biodiversity/species.csv")

# Remove unwanted empty column
species = species.loc[:, ~species.columns.str.contains("^Unnamed")]

# Fill missing values in text columns
species["Order"] = species["Order"].fillna("Unknown")
species["Family"] = species["Family"].fillna("Unknown")
species["Common Names"] = species["Common Names"].fillna("Unknown")
species["Record Status"] = species["Record Status"].fillna("Unknown")
species["Occurrence"] = species["Occurrence"].fillna("Unknown")
species["Nativeness"] = species["Nativeness"].fillna("Unknown")
species["Abundance"] = species["Abundance"].fillna("Unknown")
species["Seasonality"] = species["Seasonality"].fillna("Unknown")
species["Conservation Status"] = species["Conservation Status"].fillna("Not Listed")

# Display missing values
print("========== Missing Values After Cleaning ==========\n")
print(species.isnull().sum())

# Save the cleaned dataset
species.to_csv("Biodiversity/species_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully as 'species_cleaned.csv'")