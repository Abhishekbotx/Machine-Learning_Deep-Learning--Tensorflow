from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#--------LOGISTIC REGRESSION ---------

iris = load_iris()
X = iris.data
y = iris.target

lr_model = LogisticRegression(max_iter=500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model.fit(X_train, y_train)

pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))


