import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === 1. Load the original data ===
df = pd.read_excel("Dataset_Test.xlsx")
colonne_scelte = [3, 10, 20, 21, 23, 37, 40, 42]
df2 = df.iloc[:, colonne_scelte]

# === 2. Standardize the selected features ===
scaler = StandardScaler()
X_std = scaler.fit_transform(df2)

# === 3. Load PCA components (eigenvectors) ===
components = pd.read_csv("eigenvectors.csv").to_numpy()

# === 4. Project the standardized data into PCA space ===
pca_projected = X_std @ components.T
pca_df = pd.DataFrame(pca_projected, columns=["PC1", "PC2"])

# === 5. Load cluster labels ===
cluster = pd.read_excel("Data_Clustering.xlsx")
pca_df["Cluster"] = cluster.iloc[:, 0]

# === 6. Plot clusters: plot No Risk first, then Risk ===
plt.figure(figsize=(8, 6))

# Plot original No Risk (blue)
no_risk = pca_df[pca_df["Cluster"] == 0]
plt.scatter(no_risk["PC1"], no_risk["PC2"], c='blue', label='No Risk', alpha=0.7)

# Plot ambiguous/test-excluded subjects (black)
test = pca_df[pca_df["Cluster"] == 4]
plt.scatter(test["PC1"], test["PC2"], c='black', label='No Risk', alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Plot original Risk (red)
risk = pca_df[pca_df["Cluster"] == 1]
plt.scatter(risk["PC1"], risk["PC2"], c='red', label='Risk', alpha=0.7, edgecolors='white', linewidths=0.6, s=60)

# Plot Test No Risk (green)
no_risk_test = pca_df[pca_df["Cluster"] == 2]
plt.scatter(no_risk_test["PC1"], no_risk_test["PC2"], c='green', label='No Risk (Test)', alpha=0.9, edgecolors='white', linewidths=0.6, s=60)

# Plot Test Risk (yellow)
risk_test = pca_df[pca_df["Cluster"] == 3]
plt.scatter(risk_test["PC1"], risk_test["PC2"], c='yellow', label='Risk (Test)', alpha=0.9, edgecolors='white', linewidths=0.6, s=60)


# === 7. Axis labels, legend, and final plot settings ===
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure.png", dpi=500)
plt.show()