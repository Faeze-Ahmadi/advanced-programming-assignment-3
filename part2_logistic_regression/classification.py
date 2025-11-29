# Part 2 - Classification on Iris Dataset using Logistic Regression

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1) Load and prepare dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

X = df.drop("target", axis=1)
y = df["target"]

# 2) Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Create and train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# 4) Predict
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# 5) Scatter plot (Predicted Classes)
plt.figure(figsize=(7, 5))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap="coolwarm", s=60, edgecolor='k')
plt.title("Predicted Classes (Iris Dataset)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.grid(True)
plt.savefig("predicted_classes.png")
plt.show()

# 6) Scatter plot (Real Labels)
plt.figure(figsize=(7, 5))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap="viridis", s=60, edgecolor='k')
plt.title("Real Classes (Iris Dataset)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.grid(True)
plt.savefig("real_classes.png")
plt.show()

# 7) Confusion Matrix (more professional)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("confusion_matrix.png")
plt.show()
