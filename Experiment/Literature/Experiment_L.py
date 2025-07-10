import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# === 1. Load original dataset ===
df = pd.read_excel("Dataset.xlsx")
colonne_scelte = [3, 10, 20, 21, 23, 37, 40, 42]
df2 = df.iloc[:, colonne_scelte]

# === 2. Standardize the selected features ===
scaler = StandardScaler()
X_std = scaler.fit_transform(df2)

# === 3. Load principal components (eigenvectors) ===
components = pd.read_csv("autovettori.csv").to_numpy()

# === 4. PCA projection ===
pca_projected = X_std @ components.T
pca_df = pd.DataFrame(pca_projected, columns=["PC1", "PC2"])

# === 5. Load cluster labels ===
cluster = pd.read_excel("Dataset_Labels.xlsx")
pca_df["Cluster"] = cluster.iloc[:, 0]

# === 6. Scatter plot: first No Risk, then all other clusters ===
plt.figure(figsize=(8, 6))

# Plot No Risk individuals (Cluster 0)
no_risk = pca_df[pca_df["Cluster"] == 0]
plt.scatter(no_risk["PC1"], no_risk["PC2"], c='#1f77b4', label='1', alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
risk = pca_df[pca_df["Cluster"] == 2]
plt.scatter(risk["PC1"], risk["PC2"], c='#ff7f0e', label='2', alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 3]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#2ca02c', label='3', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 4]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#d62728', label='4', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 5]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#9467bd', label='5', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 6]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#8c564b', label='6', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 7]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#e377c2', label='7', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 8]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#7f7f7f', label='8', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 9]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#bcbd22', label='9', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# Esclusi (2 - giallo)
esclusi = pca_df[pca_df["Cluster"] == 10]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#17becf', label='10', alpha=1.0)


# === 7. Add labels, legend, and save figure ===
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig("Result.png", dpi=500)
plt.show()