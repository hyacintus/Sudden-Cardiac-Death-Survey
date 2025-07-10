import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === 1. Load the original dataset ===
df = pd.read_excel("Dataset.xlsx")

# === 2. Select only numerical columns (required for PCA) ===
df_numeric = df.select_dtypes(include=[np.number]).copy()

# === 3. Handle missing values: drop rows with NaN or replace with the mean (your choice) ===
df_numeric = df_numeric.dropna()

# === 4. Standardize the numerical data ===
scaler = StandardScaler()
X_std = scaler.fit_transform(df_numeric)

# === 5. Load eigenvectors for PCA ===
components = pd.read_csv("autovettori_Letteratura_Extra.csv").to_numpy()

# === 6. Apply PCA projection ===
pca_projected = X_std @ components.T
pca_df = pd.DataFrame(pca_projected, columns=["PC1", "PC2"])
pca_df["PC2"] = -pca_df["PC2"]  # Inverte l'asse Y

# === 7. Load clustering results ===
cluster = pd.read_excel("Dataset_Labels.xlsx")

# Adjust length if some rows were dropped due to NaNs
if len(df_numeric) < len(cluster):
    cluster = cluster.iloc[df_numeric.index]

pca_df["Cluster"] = cluster.iloc[:, 1].values

# === 8. Plot clusters: No Risk, Risk, and all others ===
plt.figure(figsize=(8, 6))

# No Risk (Cluster 0) — blue
no_risk = pca_df[pca_df["Cluster"] == 0]
plt.scatter(no_risk["PC1"], no_risk["PC2"], c='#1f77b4', label='8', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

risk = pca_df[pca_df["Cluster"] == 1]
plt.scatter(risk["PC1"], risk["PC2"], c='#f04f4f', label='1', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

esclusi = pca_df[pca_df["Cluster"] == 2]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#ff7f0e', label='2', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

esclusi = pca_df[pca_df["Cluster"] == 4]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#8c564b', label='4', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

esclusi = pca_df[pca_df["Cluster"] == 5]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#2ca02c', label='5', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

esclusi = pca_df[pca_df["Cluster"] == 6]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#17becf', label='6', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

esclusi = pca_df[pca_df["Cluster"] == 7]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#7f7f7f', label='7', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

esclusi = pca_df[pca_df["Cluster"] == 9]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='purple', label='9', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

test = pca_df[pca_df["Cluster"] == 11]
plt.scatter(test["PC1"], test["PC2"], c='#e377c2', label='11', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

test = pca_df[pca_df["Cluster"] == 10]
plt.scatter(test["PC1"], test["PC2"], c='#9467bd', label='10', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

esclusi = pca_df[pca_df["Cluster"] == 3]
plt.scatter(esclusi["PC1"], esclusi["PC2"], c='#bcbd22', label='3', alpha=1.0, edgecolors='white', linewidths=0.6, s=60)

# === 9. Axis labels and ordered legend ===
plt.xlabel("PC1")
plt.ylabel("PC2")
# Manually order legend
handles, labels = plt.gca().get_legend_handles_labels()

# Ordine desiderato (sostituisci con l’ordine che vuoi)
ordine_labels = ['8', '1', '2', '3', '4', '5', '6', '7', '9', '10', '11']

# Crea una lista ordinata di (handle, label) in base all’ordine voluto
handles_ordinati = [h for l in ordine_labels for h, lbl in zip(handles, labels) if lbl == l]
labels_ordinate = [l for l in ordine_labels if l in labels]

plt.legend(handles_ordinati, labels_ordinate, title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig("Result.png", dpi=500)
plt.show()
