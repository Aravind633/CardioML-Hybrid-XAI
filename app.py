# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import tensorflow as tf
# import shap
# from streamlit_shap import st_shap

# # Load Models
# @st.cache_resource
# def load_models():
#     clinical_model = pickle.load(open('models/clinical_ensemble.pkl', 'rb'))
#     ecg_model = tf.keras.models.load_model('models/ecg_lstm_model.h5')
#     return clinical_model, ecg_model

# clinical_model, ecg_model = load_models()

# st.title("CardioML-Aware Diagnostics")
# st.markdown("### Intelligent Framework for Predictive Heart Disease Risk Analysis")

# # TABS for different inputs
# tab1, tab2 = st.tabs(["🏥 Clinical Data Analysis", "💓 ECG Signal Analysis"])

# with tab1:
#     st.header("Patient Clinical Data")
    
#     # Input Form (Customize these based on your Kaggle dataset columns)
#     col1, col2 = st.columns(2)
#     with col1:
#         age = st.number_input("Age", 20, 100, 50)
#         chol = st.number_input("Cholesterol", 100, 400, 200)
#     with col2:
#         bp = st.number_input("Resting BP", 90, 200, 120)
#         thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        
#     # ... Add other inputs (CP, Sex, etc.) ...
    
#     if st.button("Predict Clinical Risk"):
#         # Create input array (Ensure order matches training!)
#         # This is a dummy example input
#         input_data = np.array([[age, chol, bp, thalach]]) 
        
#         # Predict
#         prediction = clinical_model.predict(input_data)
#         probability = clinical_model.predict_proba(input_data)
        
#         st.success(f"Risk Prediction: {'High Risk' if prediction[0]==1 else 'Low Risk'}")
#         st.info(f"Confidence Score: {np.max(probability)*100:.2f}%")
        
#         # [cite_start]XAI Integration [cite: 71]
#         st.subheader("Explainable AI (SHAP)")
#         explainer = shap.TreeExplainer(clinical_model)
#         shap_values = explainer.shap_values(input_data)
#         st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_data))

# with tab2:
#     st.header("ECG Signal Analysis")
#     uploaded_file = st.file_uploader("Upload ECG CSV File", type=["csv"])
    
#     if uploaded_file is not None:
#         # Process the uploaded CSV
#         ecg_data = pd.read_csv(uploaded_file, header=None)
#         ecg_signal = ecg_data.iloc[0, :187].values # Take first row, first 187 points
        
#         # Reshape for Model
#         ecg_input = ecg_signal.reshape(1, 187, 1)
        
#         # Predict
#         prediction = ecg_model.predict(ecg_input)
#         class_id = np.argmax(prediction)
        
#         classes = {0: 'Normal', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'} # MIT-BIH classes
#         st.warning(f"ECG Classification: {classes[class_id]}")
#         st.line_chart(ecg_signal)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CardioML Fusion", layout="wide")

# ==========================================
# 1. LOAD EVERYTHING
# ==========================================
@st.cache_resource
def load_system():
    # Stage 1: ECG Model
    ecg_model = tf.keras.models.load_model('models/ecg_lstm_model.h5')
    
    # Stage 2: Fusion Model
    with open('models/fusion_model.pkl', 'rb') as f:
        artifact = pickle.load(f)
        fusion_model = artifact['model']
        feature_names = artifact['features']
        
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return ecg_model, fusion_model, encoders, scaler, feature_names

try:
    ecg_model, fusion_model, encoders, scaler, feature_names = load_system()
except:
    st.error("Models missing. Run train_ecg.py and train_fusion.py first.")
    st.stop()

st.title("🫀 CardioML: Multi-Modal Fusion System")

# ==========================================
# 2. INPUTS
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Temporal Signal Analysis (ECG)")
    uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv"])
    
    # DEFAULT SCORE if no file (Simulate "Normal")
    ecg_risk_score = 0.1 
    
    if uploaded_file:
        df_ecg = pd.read_csv(uploaded_file, header=None)
        signal = df_ecg.iloc[0, :187].values
        st.line_chart(signal)
        
        # Stage 1 Prediction
        signal_reshaped = signal.reshape(1, 187, 1)
        probs = ecg_model.predict(signal_reshaped)[0]
        
        # We treat "Class 0" as Normal, Classes 1-4 as Risk
        # Risk Score = Sum of probabilities of Arrhythmia classes
        ecg_risk_score = float(np.sum(probs[1:])) 
        
        st.metric("Deep Learning Signal Score", f"{ecg_risk_score:.2f}")

with col2:
    st.subheader("2. Clinical Data")
    age = st.number_input("Age", 20, 100, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.number_input("Resting BP", 90, 200, 140)
    chol = st.number_input("Cholesterol", 120, 500, 240)
    cp_type = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    ex_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
    smoking = st.selectbox("Smoker", ["No", "Yes"])
    activity = st.slider("Physical Activity (hr/wk)", 0.0, 10.0, 1.0)
    family = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 3. FUSION PREDICTION
# ==========================================
if st.button("Analyze Combined Risk"):
    # Prepare Data
    input_data = {
        'Age': age, 'Gender': gender, 'Family History': family,
        'Resting BP': bp, 'Cholesterol': chol, 'Chest Pain Type': cp_type,
        'Exercise Induced Angina': ex_angina, 'Smoking': smoking,
        'Physical Activity (hr/wk)': activity,
        'ECG_Risk_Score': ecg_risk_score # <--- THE FUSION
    }
    
    df_in = pd.DataFrame([input_data])
    
    # Encode
    for col, le in encoders.items():
        if col in df_in.columns:
            val = df_in[col].iloc[0]
            df_in[col] = le.transform([val]) if val in le.classes_ else 0
            
    # Reorder
    df_in = df_in[feature_names]
    
    # Scale
    X_scaled = scaler.transform(df_in)
    
    # Predict
    prob = fusion_model.predict_proba(X_scaled)[0][1]
    
    st.divider()
    st.markdown(f"### Final Integrated Risk: {prob:.1%}")
    
    if prob > 0.5:
        st.error("HIGH RISK DETECTED")
    else:
        st.success("LOW RISK PROFILE")

    # ==========================================
    # 4. EXPLAINABLE AI (SHAP)
    # ==========================================
    st.subheader("3. XAI: Why this prediction?")
    
    # Use TreeExplainer for the Random Forest part of the ensemble (simplified)
    # We grab the RF estimator from the VotingClassifier
    rf_model = fusion_model.estimators_[0] 
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Visualization
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots()
    # Force plot shows exactly which features pushed risk higher (Red) or lower (Blue)
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], df_in.iloc[0], matplotlib=True, show=False)
    st.pyplot(fig, bbox_inches='tight')
    
    st.info("Red bars push risk HIGHER. Blue bars push risk LOWER.")
    st.write(f"Notice how **ECG_Risk_Score** (Value: {ecg_risk_score:.2f}) impacts the final result.")