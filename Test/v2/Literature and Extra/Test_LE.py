import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === 1. Load unified dataset ===
df = pd.read_excel("Dataset_LE.xlsx")

# === 2. Separate cluster and features (Cluster è la prima colonna) ===
y = df.iloc[:, 0]     # Cluster
X = df.iloc[:, 1:]    # features

# === 3. Standardize data ===
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# === 4. Load PCA eigenvectors ===
components = pd.read_csv("eigenvectors_LE.csv").to_numpy()

# === 5. Apply PCA projection ===
pca_projected = X_std @ components.T
pca_df = pd.DataFrame(pca_projected, columns=["PC1", "PC2"])

# Inversione asse Y
pca_df["PC2"] = -pca_df["PC2"]

# === 6. Add cluster ===
pca_df["Cluster"] = y.values

# === 7. Plot ===
plt.figure(figsize=(8, 6))

# Cluster 0 - No Risk
no_risk = pca_df[pca_df["Cluster"] == 0]
plt.scatter(no_risk["PC1"], no_risk["PC2"],
            c='blue', label='No Risk',
            alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Cluster 2 - Ambiguous / excluded
esclusi = pca_df[pca_df["Cluster"] == 2]
plt.scatter(esclusi["PC1"], esclusi["PC2"],
            c='black', label='No Risk',
            alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Cluster 1 - Risk
risk = pca_df[pca_df["Cluster"] == 1]
plt.scatter(risk["PC1"], risk["PC2"],
            c='red', label='Risk',
            alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Cluster 3 - New
new = pca_df[pca_df["Cluster"] == 3]
plt.scatter(new["PC1"], new["PC2"],
            c='yellow',
            label='New',
            alpha=0.9,
            edgecolors='black',
            linewidths=1.0,
            s=60)

# === 8. Layout ===
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
# plt.savefig("Figure_Letteratura.png", dpi=500)
plt.show()