import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("parkinsons.csv")

# Prepare the data
X = data.drop(columns=['name', 'status'], axis=1)
y = data['status']

# Select a subset of features
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2']
X = X[selected_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the best model (Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

print(f'Training Accuracy: {rf_train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {rf_test_accuracy * 100:.2f}%')

# Save the model
filename = 'parkinsons_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))

# Function to make predictions
def predict_parkinsons(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = rf_model.predict(input_data_as_numpy_array)
    return "The person has Parkinson's Disease" if prediction[0] == 1 else "The person does not have Parkinson's Disease"

# Example prediction
input_data = (197.07600, 206.89600, 192.05500, 21.033, 0.414783, 0.815285, -4.813031, 0.266482)

print(predict_parkinsons(input_data))
