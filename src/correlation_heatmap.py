import pandas as pd
import matplotlib.pyplot as plt

# Load Climate Change Dataset
climate = pd.read_csv("climate change dataset/climate_change_dataset.csv")

# Select only numeric columns
numeric_data = climate.select_dtypes(include=["number"])

# Calculate correlation
correlation = numeric_data.corr()

# Create figure
plt.figure(figsize=(10,8))

# Display heatmap
plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")

# Add color bar
plt.colorbar(label="Correlation")

# Axis labels
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)

# Add correlation values
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black"
        )

plt.title("Correlation Heatmap of Climate Change Dataset", fontsize=16)
plt.tight_layout()

plt.show()