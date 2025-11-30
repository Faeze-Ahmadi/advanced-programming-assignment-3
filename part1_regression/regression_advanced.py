import os
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

class HousingRegressor:
    """
    A full OOP wrapper around Linear Regression
    with safe loading, training, evaluation and visualization.
    """

    def __init__(self, save_dir="part1_regression"):
        self.model = LinearRegression()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.info("Initialized HousingRegressor.")

    def load_data(self):
        """Loads the California Housing Dataset."""
        try:
            data = fetch_california_housing()
            X, y = data.data, data.target
            self.feature_names = data.feature_names
            logging.info(f"Dataset loaded successfully with shape {X.shape}.")
            return X, y
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def split_data(self, X, y, test_size=0.2):
        """Split the dataset safely."""
        try:
            return train_test_split(X, y, test_size=test_size, random_state=42)
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise

    def train(self, X_train, y_train):
        """Trains the linear regression model."""
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model trained successfully.")
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise

    def evaluate(self, X_test, y_test):
        """Evaluate model performance safely."""
        try:
            score = self.model.score(X_test, y_test)
            logging.info(f"Model R^2 Score: {score:.4f}")
            return score
        except NotFittedError:
            logging.error("Model must be trained before evaluation.")
            raise

    def predict(self, X_test):
        """Predict values with error handling."""
        try:
            return self.model.predict(X_test)
        except NotFittedError:
            logging.error("Model is not trained yet.")
            raise

    def save_model(self):
        """Save trained model to .pkl file."""
        path = os.path.join(self.save_dir, "regression_model.pkl")
        joblib.dump(self.model, path)
        logging.info(f"Model saved to {path}")

    def plot_results(self, y_true, y_pred):
        """Plot true vs predicted values with a high-quality style."""
        plt.figure(figsize=(7, 7))
        plt.scatter(
            y_true, y_pred,
            c=y_pred,
            cmap="coolwarm",
            alpha=0.7,
            edgecolor="k"
        )
        plt.colorbar(label="Predicted Price")
        plt.xlabel("True Prices")
        plt.ylabel("Predicted Prices")
        plt.title("True vs Predicted House Prices (Advanced)")
        plt.grid(alpha=0.3)

        # Best-fit line
        coeffs = np.polyfit(y_true, y_pred, 1)
        line = np.poly1d(coeffs)
        plt.plot(y_true, line(y_true), color="black", linestyle="--")

        # Save
        output_path = os.path.join(self.save_dir, "regression_advanced_plot.png")
        plt.savefig(output_path, dpi=300)
        plt.show()

        logging.info(f"Advanced plot saved to {output_path}")


# ------------------------------
# Main execution (clean code)
# ------------------------------
if __name__ == "__main__":

    reg = HousingRegressor()

    # Load + split
    X, y = reg.load_data()
    X_train, X_test, y_train, y_test = reg.split_data(X, y)

    # Train & evaluate
    reg.train(X_train, y_train)
    score = reg.evaluate(X_test, y_test)

    # Predict
    y_pred = reg.predict(X_test)

    # Save model + plot
    reg.save_model()
    reg.plot_results(y_test, y_pred)
