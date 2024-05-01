# File 3: Random Forest Classifier for Hate Speech Detection

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Training.train_test import X_train, X_test, y_train, y_test

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print("Random Forest Classifier Accuracy:", accuracy)
