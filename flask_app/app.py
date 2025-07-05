import streamlit as st
import pickle
import numpy as np
import pandas as pd # You'll need this for eye disease features
import os # To check for model files

# --- Configuration for Streamlit App ---
st.set_page_config(
    page_title="Smart Health Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Define Precautions Dictionary ---
# This was in your original Flask app.py
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
        "Manage stress through meditation and relaxation techniques.",
        "Quit smoking and limit alcohol consumption.",
        "Regular check-ups and monitoring of blood pressure and cholesterol.",
        "MEDICINES:",
        "Aspirin – To prevent blood clots.",
        "Statins – To lower cholesterol.",
        "Beta-blockers – To slow heart rate and lower blood pressure."
    ],
    "kidney_disease": [
        "Control blood pressure and blood sugar levels.",
        "Reduce salt intake and follow a low-protein diet.",
        "Avoid NSAIDs (non-steroidal anti-inflammatory drugs).",
        "Stay well-hydrated.",
        "Regularly check kidney function (GFR and creatinine).",
        "MEDICINES:",
        "ACE inhibitors/ARBs – To control blood pressure and protect kidneys.",
        "Diuretics – To manage fluid retention.",
        "Sodium bicarbonate – To correct acidosis."
    ],
    "parkinsons_disease": [
        "Engage in regular physical therapy and exercise.",
        "Maintain a healthy and balanced diet.",
        "Practice speech therapy to improve communication.",
        "Use assistive devices as needed to maintain independence.",
        "Manage stress and ensure adequate rest.",
        "MEDICINES:",
        "Levodopa – Most effective medicine for Parkinson's symptoms.",
        "Dopamine agonists – Mimic the effects of dopamine in the brain.",
        "MAO-B inhibitors – Prevent the breakdown of dopamine."
    ],
    "eye_disease": [
        "Limit screen time and take regular breaks (20-20-20 rule).",
        "Ensure proper lighting while working or reading.",
        "Wear protective eyewear when exposed to harsh environments.",
        "Maintain good hydration by drinking plenty of water.",
        "Include Omega-3 rich foods in your diet (fish, flaxseed).",
        "Use lubricating eye drops if advised by a doctor.",
        "MEDICINES:",
        "Artificial tears – For symptomatic relief of dryness.",
        "Cyclosporine ophthalmic emulsion (Restasis) – To increase natural tear production.",
        "Lifitegrast ophthalmic solution (Xiidra) – To reduce inflammation."
    ]
}

# --- Load Models (Streamlit way: use st.cache_resource for efficiency) ---
# st.cache_resource ensures models are loaded only once, even if app reruns
@st.cache_resource
def load_models():
    models = {}
    try:
        models['diabetes_model'] = pickle.load(open('diabetes_model.sav', 'rb'))
        models['heart_model'] = pickle.load(open('heart_disease_model.sav', 'rb'))
        # Kidney model has scaler as well
        models['kidney_model'], models['kidney_scaler'] = pickle.load(open('kidney_disease_model.sav', 'rb'))
        models['parkinsons_model'] = pickle.load(open('parkinsons_model.sav', 'rb'))
        models['eye_model'] = pickle.load(open('eye_disease_model.sav', 'rb'))
        st.success("All models loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}. Make sure all .sav files are in the same directory.")
        st.stop() # Stop the app if models aren't found
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()
    return models

# Load all models at the start
models = load_models()
diabetes_model = models.get('diabetes_model')
heart_model = models.get('heart_model')
kidney_model = models.get('kidney_model')
kidney_scaler = models.get('kidney_scaler') # Make sure to get the scaler for kidney
parkinsons_model = models.get('parkinsons_model')
eye_model = models.get('eye_model')


# --- Sidebar Navigation ---
st.sidebar.title("Health Assistant Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Kidney Disease Prediction",
    "Parkinson's Prediction",
    "Eye Disease Prediction"
])

# --- Main Page Content ---

if page == "Home":
    st.title("Welcome to Smart Health Assistant 🧑‍⚕️")
    st.write("""
        This application helps in predicting the likelihood of various diseases based on your input.
        Please navigate through the sidebar to select a disease prediction.
    """)
    st.image("https://images.unsplash.com/photo-1576091160550-fd428edc755e?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
             caption="Health is Wealth", use_column_width=True)
    st.markdown("""
    **Disclaimer:** This tool is for informational purposes only and should not be considered medical advice.
    Always consult with a qualified healthcare professional for any health concerns.
    """)

elif page == "Diabetes Prediction":
    st.title("Diabetes Prediction 💉")
    st.write("Please enter the following details to predict Diabetes:")

    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
        glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=120)
        bloodpressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
    with col2:
        skinthickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
        insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=79)
        bmi = st.number_input('BMI Value', min_value=0.0, max_value=70.0, value=30.0, format="%.2f")
    with col3:
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input('Age', min_value=0, max_value=120, value=30)

    # Convert inputs to a numpy array for prediction
    input_data = np.array([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)

    if st.button('Predict Diabetes'):
        if diabetes_model:
            try:
                prediction = diabetes_model.predict(input_data)
                if prediction[0] == 1:
                    st.error("Prediction: The person is likely to have Diabetes.")
                    st.subheader("Precautions:")
                    for p in precautions["diabetes"]:
                        st.markdown(f"- {p}")
                else:
                    st.success("Prediction: The person is NOT likely to have Diabetes.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Diabetes model not loaded.")


elif page == "Heart Disease Prediction":
    st.title("Heart Disease Prediction ❤️")
    st.write("Please enter the following details to predict Heart Disease:")

    # Features: 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca'
    col1, col2 = st.columns(2)
    with col1:
        cp = st.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3], format_func=lambda x: {0:'Typical Angina', 1:'Atypical Angina', 2:'Non-anginal Pain', 3:'Asymptomatic'}[x])
        thalach = st.number_input('Max Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
        exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1], format_func=lambda x: {0:'No', 1:'Yes'}[x])
    with col2:
        oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=7.0, value=1.0, format="%.2f")
        slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', options=[0, 1, 2], format_func=lambda x: {0:'Upsloping', 1:'Flat', 2:'Downsloping'}[x])
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', options=[0, 1, 2, 3, 4], format_func=lambda x: {0:'0 Vessels', 1:'1 Vessel', 2:'2 Vessels', 3:'3 Vessels', 4:'Unknown'}[x])

    input_data = np.array([cp, thalach, exang, oldpeak, slope, ca]).reshape(1, -1)

    if st.button('Predict Heart Disease'):
        if heart_model:
            try:
                prediction = heart_model.predict(input_data)
                if prediction[0] == 1:
                    st.error("Prediction: The person is likely to have Heart Disease.")
                    st.subheader("Precautions:")
                    for p in precautions["heart_disease"]:
                        st.markdown(f"- {p}")
                else:
                    st.success("Prediction: The person is NOT likely to have Heart Disease.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Heart disease model not loaded.")

elif page == "Kidney Disease Prediction":
    st.title("Kidney Disease Prediction 🩸")
    st.write("Please enter the following details to predict Kidney Disease:")

    # Features: 'sg', 'al', 'hemo', 'pcv', 'htn', 'dm'
    col1, col2, col3 = st.columns(3)
    with col1:
        sg = st.number_input('Specific Gravity (sg)', min_value=1.000, max_value=1.030, value=1.015, format="%.3f")
        al = st.number_input('Albumin (al)', min_value=0, max_value=5, value=0)
    with col2:
        hemo = st.number_input('Hemoglobin (hemo)', min_value=0.0, max_value=20.0, value=12.0, format="%.2f")
        pcv = st.number_input('Packed Cell Volume (pcv)', min_value=0, max_value=60, value=40)
    with col3:
        htn = st.selectbox('Hypertension (htn)', options=[0, 1], format_func=lambda x: {0:'No', 1:'Yes'}[x])
        dm = st.selectbox('Diabetes Mellitus (dm)', options=[0, 1], format_func=lambda x: {0:'No', 1:'Yes'}[x])

    input_data = np.array([sg, al, hemo, pcv, htn, dm]).reshape(1, -1)

    if st.button('Predict Kidney Disease'):
        if kidney_model and kidney_scaler: # Ensure both are loaded
            try:
                # Scale the input features using the pre-trained scaler
                scaled_input_data = kidney_scaler.transform(input_data)
                prediction = kidney_model.predict(scaled_input_data)
                if prediction[0] == 1: # Assuming 1 is for CKD (Chronic Kidney Disease)
                    st.error("Prediction: The person is likely to have Kidney Disease (CKD).")
                    st.subheader("Precautions:")
                    for p in precautions["kidney_disease"]:
                        st.markdown(f"- {p}")
                else:
                    st.success("Prediction: The person is NOT likely to have Kidney Disease.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Kidney model or scaler not loaded.")

elif page == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction 🚶")
    st.write("Please enter the following details to predict Parkinson's Disease:")

    # Features: 'fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP',
    #           'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'MDVP_APQ', 'MDVP_RAP',
    #           'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    # There are 22 features here. We should use st.columns or st.expander for better UI.
    st.markdown("Enter voice parameters (refer to Parkinson's disease datasets for typical ranges):")

    input_features_parkinsons = []
    # Using columns for better layout
    cols = st.columns(3)
    labels = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
        'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',
        'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2',
        'D2', 'PPE'
    ]
    default_values = [
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.0039, 0.01109,
        0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
        0.414735, 0.778665, -4.813031, 0.266482, 2.301442, 0.284654, 1.621456, 0.237251
    ] # Example median values for initial display

    for i, label in enumerate(labels):
        with cols[i % 3]:
            # Use a generic number_input, assuming most are floats. Adjust `format` if integer.
            val = st.number_input(label, value=default_values[i], format="%.5f")
            input_features_parkinsons.append(val)

    parkinsons_input_data = np.array(input_features_parkinsons).reshape(1, -1)

    if st.button("Predict Parkinson's"):
        if parkinsons_model:
            try:
                prediction = parkinsons_model.predict(parkinsons_input_data)
                if prediction[0] == 1:
                    st.error("Prediction: The person is likely to have Parkinson's Disease.")
                    st.subheader("Precautions:")
                    for p in precautions["parkinsons_disease"]:
                        st.markdown(f"- {p}")
                else:
                    st.success("Prediction: The person is NOT likely to have Parkinson's Disease.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Parkinson's model not loaded.")

elif page == "Eye Disease Prediction":
    st.title("Dry Eye Disease Prediction 👁️")
    st.write("Please enter the following details to predict Dry Eye Disease:")

    # Features based on your eye_model.py:
    # "Physical activity", "Average screen time", "Sleep duration", "Systolic BP", "Weight"
    # Note: Your eye_model.py also processes 'Blood pressure' into 'Systolic BP' and 'Diastolic BP'.
    # We will need to capture Systolic BP directly here as the model expects it.
    # It also processes other binary features and Gender, but the 'selected_features' for prediction
    # are limited to the 5 mentioned. Ensure your 'eye_model.sav' uses these 5 features for prediction
    # or adjust the features collected here accordingly.

    col1, col2 = st.columns(2)
    with col1:
        physical_activity = st.number_input('Physical activity (hours/week)', min_value=0.0, max_value=100.0, value=5.0, format="%.2f")
        average_screen_time = st.number_input('Average screen time (hours/day)', min_value=0.0, max_value=24.0, value=6.0, format="%.2f")
        sleep_duration = st.number_input('Sleep duration (hours/night)', min_value=0.0, max_value=15.0, value=7.0, format="%.2f")
    with col2:
        systolic_bp = st.number_input('Systolic Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)
        weight = st.number_input('Weight (kg)', min_value=20.0, max_value=300.0, value=70.0, format="%.2f")

    # The features must be in the exact order as expected by your eye_model.
    # Based on your eye_model.py: selected_features = ["Physical activity", "Average screen time", "Sleep duration", "Systolic BP", "Weight"]
    input_data_eye = np.array([
        physical_activity,
        average_screen_time,
        sleep_duration,
        systolic_bp,
        weight
    ]).reshape(1, -1)


    if st.button('Predict Dry Eye Disease'):
        if eye_model:
            try:
                prediction = eye_model.predict(input_data_eye)
                probability = eye_model.predict_proba(input_data_eye)[0][1] # Probability of class 1 (Dry Eye Disease)

                if prediction[0] == 1:
                    st.error(f"Prediction: Dry Eye Disease Detected. Risk Probability: {probability:.2f}")
                    st.subheader("Precautions:")
                    for p in precautions["eye_disease"]:
                        st.markdown(f"- {p}")
                else:
                    st.success(f"Prediction: No Dry Eye Disease Detected. Risk Probability: {probability:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Eye disease model not loaded.")
