import os
import logging
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


class IrisClassifier:
    """Advanced Logistic Regression classifier with visualization,
       PCA support, error handling, and model persistence."""

    def __init__(self, save_dir="part2_logistic_regression"):
        self.model = LogisticRegression(max_iter=500)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.info("Initialized IrisClassifier.")

    def load_data(self):
        """Load Iris dataset into DataFrame."""
        try:
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df["target"] = iris.target
            self.feature_names = iris.feature_names
            self.target_names = iris.target_names
            logging.info("Dataset loaded successfully.")
            return df
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def preprocess(self, df):
        """Split & scale dataset."""
        try:
            X = df.drop("target", axis=1)
            y = df["target"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.25, random_state=42, stratify=y
            )

            self.scaler = scaler
            logging.info("Preprocessing completed (scaling + splitting).")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Preprocessing error: {e}")
            raise

    def train(self, X_train, y_train):
        """Train logistic regression."""
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model trained successfully.")
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise

    def evaluate(self, X_test, y_test):
        """Return model accuracy."""
        try:
            accuracy = self.model.score(X_test, y_test)
            logging.info(f"Accuracy: {accuracy:.4f}")
            return accuracy
        except NotFittedError:
            logging.error("Cannot evaluate before training.")
            raise

    def predict(self, X_test):
        try:
            return self.model.predict(X_test)
        except NotFittedError:
            logging.error("Model not trained yet.")
            raise

    def save_model(self):
        path = os.path.join(self.save_dir, "iris_model.pkl")
        joblib.dump(self.model, path)
        logging.info(f"Model saved to {path}")

    def plot_pca(self, X_test, y_test, y_pred):
        """Visualize in 2D using PCA."""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)

        # Predicted classes
        plt.figure(figsize=(7, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="coolwarm", s=70, edgecolor="k")
        plt.title("Predicted Classes (PCA Projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, "predicted_pca.png"), dpi=300)
        plt.show()

        # Actual classes
        plt.figure(figsize=(7, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap="viridis", s=70, edgecolor="k")
        plt.title("Real Classes (PCA Projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, "real_pca.png"), dpi=300)
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred):
        """Styled confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.target_names)

        plt.figure(figsize=(6, 6))
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - Logistic Regression")
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix_advanced.png"), dpi=300)
        plt.show()


if __name__ == "__main__":

    clf = IrisClassifier()

    df = clf.load_data()
    X_train, X_test, y_train, y_test = clf.preprocess(df)
    clf.train(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = clf.evaluate(X_test, y_test)

    clf.save_model()
    clf.plot_pca(X_test, y_test, y_pred)
    clf.plot_confusion_matrix(y_test, y_pred)
