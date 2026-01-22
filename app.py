import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import shap
from streamlit_shap import st_shap

# Load Models
@st.cache_resource
def load_models():
    clinical_model = pickle.load(open('models/clinical_ensemble.pkl', 'rb'))
    ecg_model = tf.keras.models.load_model('models/ecg_lstm_model.h5')
    return clinical_model, ecg_model

clinical_model, ecg_model = load_models()

st.title("CardioML-Aware Diagnostics")
st.markdown("### Intelligent Framework for Predictive Heart Disease Risk Analysis")

# TABS for different inputs
tab1, tab2 = st.tabs(["🏥 Clinical Data Analysis", "💓 ECG Signal Analysis"])

with tab1:
    st.header("Patient Clinical Data")
    
    # Input Form (Customize these based on your Kaggle dataset columns)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        chol = st.number_input("Cholesterol", 100, 400, 200)
    with col2:
        bp = st.number_input("Resting BP", 90, 200, 120)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        
    # ... Add other inputs (CP, Sex, etc.) ...
    
    if st.button("Predict Clinical Risk"):
        # Create input array (Ensure order matches training!)
        # This is a dummy example input
        input_data = np.array([[age, chol, bp, thalach]]) 
        
        # Predict
        prediction = clinical_model.predict(input_data)
        probability = clinical_model.predict_proba(input_data)
        
        st.success(f"Risk Prediction: {'High Risk' if prediction[0]==1 else 'Low Risk'}")
        st.info(f"Confidence Score: {np.max(probability)*100:.2f}%")
        
        # [cite_start]XAI Integration [cite: 71]
        st.subheader("Explainable AI (SHAP)")
        explainer = shap.TreeExplainer(clinical_model)
        shap_values = explainer.shap_values(input_data)
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_data))

with tab2:
    st.header("ECG Signal Analysis")
    uploaded_file = st.file_uploader("Upload ECG CSV File", type=["csv"])
    
    if uploaded_file is not None:
        # Process the uploaded CSV
        ecg_data = pd.read_csv(uploaded_file, header=None)
        ecg_signal = ecg_data.iloc[0, :187].values # Take first row, first 187 points
        
        # Reshape for Model
        ecg_input = ecg_signal.reshape(1, 187, 1)
        
        # Predict
        prediction = ecg_model.predict(ecg_input)
        class_id = np.argmax(prediction)
        
        classes = {0: 'Normal', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'} # MIT-BIH classes
        st.warning(f"ECG Classification: {classes[class_id]}")
        st.line_chart(ecg_signal)