import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === 1. Load the original dataset ===
df = pd.read_excel("Dataset_Trial.xlsx")

# === 2. Select only numeric columns (required for PCA) ===
df_numeric = df.select_dtypes(include=[np.number]).copy()

# === 3. Handle missing values: drop rows with NaN values or replace with column mean ===
df_numeric = df_numeric.dropna()

# === 4. Standardize the data ===
scaler = StandardScaler()
X_std = scaler.fit_transform(df_numeric)

# === 5. Load PCA components (eigenvectors) ===
components = pd.read_csv("autovettori_Letteratura_Extra.csv").to_numpy()

# === 6. Apply PCA projection ===
pca_projected = X_std @ components.T
pca_df = pd.DataFrame(pca_projected, columns=["PC1", "PC2"])
pca_df["PC2"] = -pca_df["PC2"]  # Inverte l'asse Y

# === 7. Load cluster labels: 0 = No Risk, 1 = Risk, 2 = Ambiguous, 3 = New ===
cluster = pd.read_excel("Data_Clustering.xlsx")

# Adjust clustering labels in case rows with NaNs were dropped
if len(df_numeric) < len(cluster):
    cluster = cluster.iloc[df_numeric.index]

pca_df["Cluster"] = cluster.iloc[:, 1].values

# === 8. Plot PCA results ===
plt.figure(figsize=(8, 6))

# Plot original 'No Risk' points (Cluster 0 - blue)
no_risk = pca_df[pca_df["Cluster"] == 0]
plt.scatter(no_risk["PC1"], no_risk["PC2"], c='blue', label='No Risk', alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Plot ambiguous/excluded cases (Cluster 2 - black)
esclusi = pca_df[pca_df["Cluster"] == 2]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='black', label='No Risk', alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Plot original 'Risk' points (Cluster 1 - red)
risk = pca_df[pca_df["Cluster"] == 1]
plt.scatter(risk["PC1"], risk["PC2"], c='red', label='Risk', alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Plot new tested subjects (Cluster 3 - yellow)
new = pca_df[pca_df["Cluster"] == 3]
plt.scatter(new["PC1"], new["PC2"], c='yellow', label='New', alpha=0.9, edgecolors='white', linewidths=0.6, s=60)

# === 9. Axis labels and legend ===
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure.png", dpi=500)
plt.show()
