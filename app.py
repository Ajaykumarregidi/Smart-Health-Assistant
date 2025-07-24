from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# 🔹 Load models
try:
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
    kidney_model, scaler = pickle.load(open('kidney_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
    eye_model = pickle.load(open('eye_disease_model.sav', 'rb'))
except Exception as e:
    print("Error loading models:", e)

# 🔹 Disease Precautions Dictionary
precautions = {
    "diabetes": [
        "Maintain a balanced diet low in sugar and carbohydrates.",
        "Exercise regularly for at least 30 minutes per day.",
        "Monitor blood sugar levels frequently.",
        "Stay hydrated and avoid sugary drinks.",
        "Take prescribed medications as directed by your doctor.",
        "MEDICINES:",
        "Metformin – First-line medicine for Type 2 diabetes.",
        "Insulin – For blood sugar control in severe cases.",
        "Glipizide – Stimulates insulin production."
    ],
    "heart_disease": [
        "Follow a heart-healthy diet rich in vegetables, fruits, and whole grains.",
        "Limit salt, sugar, and unhealthy fats.",
        "Engage in regular physical activity like walking or yoga.",
        "Manage stress through meditation or relaxation techniques.",
        "Avoid smoking and excessive alcohol consumption.",
        "MEDICINES:",
        "Aspirin – Prevents blood clots.",
        "Atorvastatin (Lipitor) – Lowers cholesterol",
        "Lisinopril – Controls blood pressure.",
        "Metoprolol – Reduces heart strain"
    ],
    "kidney_disease": [
        "Reduce salt intake to maintain healthy blood pressure.",
        "Drink plenty of water to support kidney function.",
        "Limit protein intake (especially red meat and processed foods).",
        "Avoid overuse of painkillers and NSAIDs.",
        "Regularly check kidney function if you have diabetes or hypertension.",
        "MEDICINES:",
        "Losartan – Protects kidneys & lowers BP.",
        "Furosemide (Lasix) – Helps remove excess fluids.",
        "Sodium Bicarbonate – Manages kidney-related acidity."
    ],
    "parkinsons": [
        "Engage in regular physical therapy to maintain mobility.",
        "Follow a nutritious diet rich in antioxidants and fiber.",
        "Stay socially active and engaged in stimulating activities.",
        "Practice relaxation techniques like yoga or meditation.",
        "Take medications as prescribed and attend regular check-ups.",
        "MEDICINES:",
        "Levodopa + Carbidopa (Sinemet) – Improves movement.",
        "Pramipexole – Helps with tremors & stiffness."

    ],  
    "eye_disease": [
        "Maintain a balanced diet low in sugar and carbohydrates.",
        "Exercise regularly for at least 30 minutes per day.",
        "Monitor blood sugar levels frequently.",
        "Stay hydrated and avoid sugary drinks.",
        "Take prescribed medications as directed by your doctor.",
    ]
    
}

# 🔹 Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

# 🔹 Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'ajay123' and password == '123456789':
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid Credentials')

    return render_template('login.html')

# 🔹 Dashboard (After Login)
@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# 🔹 Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# 🔹 Diabetes Prediction
@app.route('/predict/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            final_features = np.array([features])

            # Make prediction
            prediction = diabetes_model.predict(final_features)

            if prediction[0] == 1:
                result = "Diabetes Detected"
                advice = precautions["diabetes"]
            else:
                result = "No Diabetes"
                advice = []

        except Exception as e:
            result = f"Error in prediction: {str(e)}"
            advice = []

        return render_template('diabetes.html', result=result, precautions=advice)

    return render_template('diabetes.html')

# 🔹 Heart Disease Prediction
@app.route('/predict/heart', methods=['GET', 'POST'])
def predict_heart():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            final_features = np.array([features])

            # Make prediction
            prediction = heart_model.predict(final_features)

            if prediction[0] == 1:
                result = "Heart Disease Detected"
                advice = precautions["heart_disease"]
            else:
                result = "No Heart Disease"
                advice = []

        except Exception as e:
            result = f"Error in prediction: {str(e)}"
            advice = []

        return render_template('heart.html', result=result, precautions=advice)

    return render_template('heart.html')

# 🔹 Kidney Disease Prediction
@app.route('/predict/kidney', methods=['GET', 'POST'])
def predict_kidney():
    if request.method == 'POST':
        try:
            features = [float(x) if x.strip() else 0.0 for x in request.form.values()]
            final_features = np.array([features])

            # Scale the input features
            final_features = scaler.transform(final_features)

            # Make prediction
            prediction = kidney_model.predict(final_features)

            if prediction[0] == 1:
                result = "Chronic Kidney Disease Detected"
                advice = precautions["kidney_disease"]
            else:
                result = "No Kidney Disease"
                advice = []

        except Exception as e:
            result = f"Error in prediction: {str(e)}"
            advice = []

        return render_template('kidney.html', result=result, precautions=advice)

    return render_template('kidney.html')

# 🔹 Parkinson's Disease Prediction
@app.route('/predict/parkinsons', methods=['GET', 'POST'])
def predict_parkinsons():
    if request.method == 'POST':
        try:
            features = [float(x) if x.strip() else 0.0 for x in request.form.values()]
            final_features = np.array([features])

            # Make prediction
            prediction = parkinsons_model.predict(final_features)

            if prediction[0] == 1:
                result = "Parkinson's Disease Detected"
                advice = precautions["parkinsons"]
            else:
                result = "No Parkinson's Disease"
                advice = []

        except Exception as e:
            result = f"Error in prediction: {str(e)}"
            advice = []

        return render_template('parkinsons.html', result=result, precautions=advice)

    return render_template('parkinsons.html')

# 🔹 Eye Disease Prediction
@app.route('/predict/eye', methods=['GET', 'POST'])
def predict_eye():
    if request.method == 'POST':
        try:
            features = [float(x) if x.strip() else 0.0 for x in request.form.values()]
            final_features = np.array([features]).reshape(1, -1)  # Reshape for a single sample
            
            # Make prediction
            prediction = eye_model.predict(final_features)
            probability = eye_model.predict_proba(final_features)[0][1]  # Get probability of class 1
            
            if prediction[0] == 1:
                result = "Dry Eye Disease Detected"
                advice = precautions["eye_disease"]
            else:
                result = "No Dry Eye Disease"
                advice = []
            
        except Exception as e:
            result = f"Error in prediction: {str(e)}"
            advice = []
            probability = None
        
        return render_template(
            'eye.html',
            result=result,
            probability=f"Risk Probability: {probability:.2f}" if probability is not None else "",
            precautions=advice
        )
    
    return render_template('eye.html')




# 🔹 Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
