import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# =====================================================
# 🧩 PART 1 — SIMPLE LINEAR REGRESSION (Hours vs Marks)
# =====================================================
print("\n========== SIMPLE LINEAR REGRESSION ==========\n")

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
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
print("x_tesstt:", X_test)
# Predict
y_pred = model_simple.predict(X_test)

# Evaluate
print("Predictions:", y_pred)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# How much marks if someone studies 9 hours?
print("Marks for 6 hours study:", model_simple.predict([[6]])[0])
print("Marks for 9 hours study:", model_simple.predict([[9]])[0])


plt.scatter(X_train, y_train)

# Plot the regression line
predicted_line = model_simple.predict(X_train)
plt.plot(X_train, model_simple.predict(X_train), color="orange", label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression (Hours vs Marks)")
plt.legend()
plt.grid(True)
plt.show()

# =======================================================
# 🧩 PART 2 — MULTIPLE LINEAR REGRESSION (2 Input Features)
# =======================================================
print("\n========== MULTIPLE LINEAR REGRESSION ==========\n")

# Load dataset
data = pd.read_csv("data/Student_Marks.csv")

print("Dataset Preview:\n", data.head())

# Features (number_courses, time_study)
X = data[["number_courses", "time_study"]]
y = data["Marks"]

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model_multi.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print("Coefficients:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)

# Visualization: Actual vs Predicted Marks (Line Plot)
y_test_plot = y_test.reset_index(drop=True)
y_pred_plot = pd.Series(y_pred)

plt.plot(y_test_plot, label="Actual Marks", marker="o")
plt.plot(y_pred_plot, label="Predicted Marks", marker="x")
plt.title("Multiple Linear Regression — Actual vs Predicted")
plt.xlabel("Test Sample Index")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================================
# 🧩 PART 3 — 3D VISUALIZATION OF MULTIPLE REGRESSION PLANE
# ==========================================================
print("\n========== 3D REGRESSION VISUALIZATION ==========\n")

# Train again on entire dataset for smoother 3D surface
X_full = data[["number_courses", "time_study"]]
y_full = data["Marks"]
model_full = LinearRegression()
model_full.fit(X_full, y_full)

# Prepare 3D Surface Grid
x_surf, y_surf = np.meshgrid(
    np.linspace(X_full.number_courses.min(), X_full.number_courses.max(), 10),
    np.linspace(X_full.time_study.min(), X_full.time_study.max(), 10)
)
z_surf = model_full.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)

# Create 3D Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Data points
ax.scatter(
    X_full["number_courses"], X_full["time_study"], y_full,
    color="blue", s=50, label="Data Points"
)

# Regression plane
ax.plot_surface(x_surf, y_surf, z_surf, color="orange", alpha=0.5, label="Regression Plane")

ax.set_xlabel("Number of Courses")
ax.set_ylabel("Study Time (Hours)")
ax.set_zlabel("Marks")
plt.title("3D Visualization — Multiple Linear Regression Plane")
plt.show()

print("\n✅ All models executed successfully!\n")