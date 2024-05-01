# File 4: Text Vectorization and Data Splitting for Hate Speech Detection

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from Training.hatespeech import x, y  # Assuming x and y are preprocessed features and labels

# Initialize CountVectorizer
cv = CountVectorizer()

# Convert text data (x) into numerical features (X)
X = cv.fit_transform(x)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
