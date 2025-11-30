import os
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


class IrisClustering:
    """Advanced KMeans clustering with PCA, evaluation,
       visualization, error handling, and model saving."""

    def __init__(self, save_dir="part3_clustering"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.info("Initialized IrisClustering.")

    def load_data(self):
        """Load dataset safely."""
        try:
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.iris = iris
            logging.info("Dataset loaded successfully.")
            return df
        except Exception as e:
            logging.error(f"Dataset load error: {e}")
            raise

    def preprocess(self, df):
        """Scale features before clustering."""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df.values)
            self.scaler = scaler
            logging.info("Feature scaling completed.")
            return X_scaled
        except Exception as e:
            logging.error(f"Preprocessing error: {e}")
            raise

    def elbow_method(self, X):
        """Plot Elbow Method to find optimal k."""
        distortions = []

        for k in range(2, 8):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(X)
            distortions.append(model.inertia_)

        plt.figure(figsize=(7, 5))
        plt.plot(range(2, 8), distortions, marker='o', linestyle='--')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for KMeans")
        plt.grid(alpha=0.3)

        path = os.path.join(self.save_dir, "elbow_method.png")
        plt.savefig(path, dpi=300)
        plt.show()
        logging.info(f"Elbow Method plot saved to {path}")

    def train(self, X, k=3):
        """Train the KMeans model."""
        try:
            self.model = KMeans(n_clusters=k, random_state=42)
            self.clusters = self.model.fit_predict(X)

            sil = silhouette_score(X, self.clusters)
            logging.info(f"KMeans trained with silhouette score: {sil:.4f}")

            return self.clusters, sil
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise

    def save_model(self):
        """Save trained KMeans model."""
        path = os.path.join(self.save_dir, "kmeans_model.pkl")
        joblib.dump(self.model, path)
        logging.info(f"KMeans model saved to {path}")

    def plot_pca_clusters(self, X, clusters, title, filename):
        """Plot clustering results in PCA space."""

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(7, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="Set1",
                    s=70, edgecolor="k", alpha=0.9)

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.3)

        path = os.path.join(self.save_dir, filename)
        plt.savefig(path, dpi=300)
        plt.show()

        logging.info(f"{title} plot saved to {path}")

    def plot_real_labels(self, X, labels):
        """Plot real iris labels (for comparison)."""

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(7, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis",
                    s=70, edgecolor="k", alpha=0.9)

        plt.title("Real Iris Classes (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.3)

        path = os.path.join(self.save_dir, "real_labels_pca.png")
        plt.savefig(path, dpi=300)
        plt.show()

        logging.info(f"Real label PCA plot saved to {path}")


if __name__ == "__main__":

    cluster = IrisClustering()

    df = cluster.load_data()
    X_scaled = cluster.preprocess(df)

    # Elbow Method (Professional)
    cluster.elbow_method(X_scaled)

    # Train with k=3 (Iris known classes)
    clusters, sil_score = cluster.train(X_scaled, k=3)
    logging.info(f"Silhouette Score: {sil_score:.4f}")

    cluster.save_model()

    # PCA visualizations
    cluster.plot_pca_clusters(X_scaled, clusters,
                              "KMeans Clusters (PCA Projection)",
                              "clusters_pca.png")

    cluster.plot_real_labels(X_scaled, cluster.iris.target)

    # Print cluster centers
    print("\nCluster Centers (scaled feature space):")
    print(cluster.model.cluster_centers_)
