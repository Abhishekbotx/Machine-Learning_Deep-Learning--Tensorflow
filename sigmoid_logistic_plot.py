import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate values for x-axis
x = np.linspace(-10, 10, 200)
y = sigmoid(x)

# Plot
plt.plot(x, y)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.show()



# Dataset
marks = np.array([20, 25, 35, 42, 50, 55, 60, 72, 85, 90]).reshape(-1, 1)
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(marks, labels)

# Generate smooth x-values for the curve
x_test = np.linspace(0, 100, 200).reshape(-1, 1)
probabilities = model.predict_proba(x_test)[:, 1]   # probability of pass (class 1)

# Plot data points
plt.scatter(marks, labels, color="blue", label="Actual Data")

# Plot logistic curve
plt.plot(x_test, probabilities, color="red", label="Logistic Curve")

# Decision boundary
boundary = -(model.intercept_[0] / model.coef_[0][0])
print("boundary::",boundary)
plt.axvline(boundary, color="green", linestyle="--", label=f"Decision Boundary = {boundary:.2f}")

plt.xlabel("Marks")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Curve")
plt.legend()
plt.grid(True)
plt.show()


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target   # 3 classes: 0,1,2

# Model
model = LogisticRegression(max_iter=500, multi_class="multinomial")
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))

# Probabilities
print("\nProbabilities for first sample:")
print(model.predict_proba([X[0]]))

print("\nClassification Report:")
print(classification_report(y, y_pred))