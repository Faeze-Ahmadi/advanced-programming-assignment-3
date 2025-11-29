# Part 3 - KMeans Clustering on Iris Dataset

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1) Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

X = df.values  # we only use features

# 2) Feature scaling (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Apply KMeans (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 4) Plot clusters (first 2 features)
plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="Set1", s=60, edgecolor='k')
plt.title("KMeans Clusters (Iris Dataset)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.grid(True)
plt.savefig("clusters.png")
plt.show()

# 5) Compare clusters with real labels
plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=iris.target, cmap="viridis", s=60, edgecolor='k')
plt.title("Real Classes (Iris Dataset)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.grid(True)
plt.savefig("real_labels.png")
plt.show()

# 6) Print cluster centers
print("\nCluster Centers (scaled feature space):")
print(kmeans.cluster_centers_)
