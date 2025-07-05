import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Dry_Eye_Dataset.csv")

# Convert binary categorical columns ('Y'/'N') to 0/1
binary_columns = [
    "Sleep disorder", "Wake up during night", "Feel sleepy during day",
    "Caffeine consumption", "Alcohol consumption", "Smoking", 
    "Medical issue", "Ongoing medication", "Smart device before bed", 
    "Blue-light filter", "Discomfort Eye-strain", "Redness in eye", 
    "Itchiness/Irritation in eye", "Dry Eye Disease"
]
df[binary_columns] = df[binary_columns].applymap(lambda x: 1 if x == 'Y' else 0)

# Convert 'Gender' to numerical
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1}).astype(int)

# Extract 'Systolic BP' and 'Diastolic BP' from 'Blood pressure'
df[['Systolic BP', 'Diastolic BP']] = df['Blood pressure'].str.split('/', expand=True).astype(int)
df.drop(columns=['Blood pressure'], inplace=True)

# Define top 5 selected features
selected_features = ["Physical activity", "Average screen time", "Sleep duration", "Systolic BP", "Weight"]
X = df[selected_features]
y = df["Dry Eye Disease"]

# Split the dataset into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gbc.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = gbc.predict(X_test)
y_prob = gbc.predict_proba(X_test)[:, 1]  # Probability of disease (class 1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("First 10 Probability Scores:", y_prob[:10])
import pickle

filename = 'eye_disease_model.sav'
pickle.dump((gbc), open(filename, 'wb'))
print("Model saved successfully!")
