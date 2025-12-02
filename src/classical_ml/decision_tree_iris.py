from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import json

# 1. Load dataset
data = load_iris()
# print("data::",data)
X = data.data       # features (measurements)sepal length, sepal width, petal length, petal width
y = data.target     # labels (flower species) encoded in 0,1,2 indexing 



# 2️. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=26 # 0.2 = 20% test data, 80% train data
)                                   # 42 gives most of the time same prediction thats why we are using 42

# print("x_train:",X_train,"X_test:",X_test,"y_train:",y_train,"y_test:",y_test)

# 3️. Train model
model = DecisionTreeClassifier() #We create a Decision Tree classifier
model.fit(X_train, y_train) #.fit() means learn patterns from training data

# 4️. Predict
y_pred = model.predict(X_test)

print("y_test:",y_test)
print("y_pred:",y_pred)

#  Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))


#  Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#  Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#  Visualize Tree
plt.figure(figsize=(10,8))
plot_tree(model, filled=True, class_names=data.target_names, feature_names=data.feature_names)
plt.show()
