import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "marks": [45, 50, 60, 65, 70, 75, 85, 90, 95, 98],
}

df = pd.DataFrame(data)

print("dataframe::", df)

# Features (X) and Labels (y)
X = df[["hours"]]
y = df["marks"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("x_tesstt:", X_test)
# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Predictions:", y_pred)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# How much marks if someone studies 9 hours?
print("Marks for 6 hours study:", model.predict([[6]])[0])
print("Marks for 9 hours study:", model.predict([[9]])[0])


plt.scatter(X_train, y_train)

# Plot the regression line
predicted_line = model.predict(X_train)
plt.plot(X_train, predicted_line)

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Study Hours vs Marks - Linear Regression")
plt.show()
