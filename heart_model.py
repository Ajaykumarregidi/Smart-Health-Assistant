import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("heart.csv")

# Select only the most important features
selected_features = ['cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']
X = data[selected_features]
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression Model
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)

# Model Evaluation
train_accuracy = accuracy_score(y_train, logistic_classifier.predict(X_train))
test_accuracy = accuracy_score(y_test, logistic_classifier.predict(X_test))

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# Example Input Data (must match selected features)
input_data = np.array([3, 150, 0, 2.3, 0, 0])  # Example values for the 6 selected features
input_data_reshaped = input_data.reshape(1, -1)

# Prediction
prediction = logistic_classifier.predict(input_data_reshaped)


if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')

# Save the reduced feature model
filename = 'heart_disease_model.sav'
pickle.dump(logistic_classifier, open(filename, 'wb'))