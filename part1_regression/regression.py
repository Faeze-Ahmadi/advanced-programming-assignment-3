# simple model of linear regressin

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1) Load dataset
data = fetch_california_housing()
X = data.data       # features
y = data.target     # target (house prices)

# 2) Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4) Predict on test set
y_pred = model.predict(X_test)

# 5) Evaluate model (R^2 score)
score = model.score(X_test, y_test)
print("Model R^2 score:", score)

# 6) Plot true vs predicted values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("True vs Predicted House Prices")
plt.grid(True)

# Save plot as PNG
plt.savefig("part1_regression/regression_plot.png")
plt.show()
