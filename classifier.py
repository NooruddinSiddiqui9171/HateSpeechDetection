# File 1: Decision Tree Classifier for Hate Speech Detection

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Training.train_test import X_train, X_test, y_train, y_test

# Initialize Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print("Decision Tree Classifier Accuracy:", accuracy)
