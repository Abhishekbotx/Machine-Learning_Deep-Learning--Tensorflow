from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Train model
marks = np.array([20, 25, 35, 42, 50, 55, 60, 72, 85, 90])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Create dataframe
df = pd.DataFrame({
    "Marks": marks,
    "Pass": labels
})

x = df[["Marks"]]
y = df["Pass"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict probabilities
y_prob = model.predict_proba(x_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# AUC
auc = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc)

# Plot ROC
plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

