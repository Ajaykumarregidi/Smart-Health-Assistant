import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔹 Load dataset
kidney_df = pd.read_csv('kidney.csv')

# 🔹 Handle missing values using most frequent strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(kidney_df), columns=kidney_df.columns)

# 🔹 Fix inconsistent values
df_imputed["classification"] = df_imputed["classification"].replace({"ckd\t": "ckd", "notckd": "not ckd"})
df_imputed["cad"] = df_imputed["cad"].replace("\tno", "no")
df_imputed["dm"] = df_imputed["dm"].replace(["\tno", "\tyes", " yes"], ["no", "yes", "yes"])
df_imputed["wc"] = df_imputed["wc"].replace(["\t6200", "\t?", "\t8400"], ["9800", "5.2", "9800"])
df_imputed["pcv"] = df_imputed["pcv"].replace(["\t43", "\t?"], ["41", "41"])
df_imputed["rc"] = df_imputed["rc"].replace("\t?", "5.2")

# 🔹 Convert categorical values to numerical
dictionary = {
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
}
df = df_imputed.replace(dictionary)

# 🔹 Encode classification column
label_encoder = LabelEncoder()
df["classification"] = label_encoder.fit_transform(df["classification"])

# 🔹 Convert numeric columns to float
for col in df.select_dtypes(exclude=["object"]).columns:
    df[col] = df[col].astype(float)

# 🔹 Select top features
selected_features = ["sg", "al", "hemo", "pcv", "htn", "dm"]
X = df[selected_features]
y = df["classification"]

# 🔹 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Standardize dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Train the model
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# 🔹 Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# 🔹 Save the model and scaler
filename = 'kidney_disease_model.sav'
pickle.dump((model, scaler), open(filename, 'wb'))
print("Model saved successfully!")
