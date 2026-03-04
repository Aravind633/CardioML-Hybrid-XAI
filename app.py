import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ---------------------------------------
# PATH FIX
# ---------------------------------------
sys.path.append(os.path.abspath("models"))
from ecg_inference import get_ecg_risk_score

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="CardioML – Explainable ECG-Aware Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------------------------------
# LOAD MODELS
# ---------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("models/fusion_pipeline.pkl")

@st.cache_resource
def load_shap():
    obj = joblib.load("models/shap_explainer.pkl")
    return obj["explainer"], obj["preprocessor"], obj["feature_names"]

fusion_model = load_pipeline()
shap_explainer, shap_preprocessor, shap_features = load_shap()

# ---------------------------------------
# HEADER
# ---------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>❤️ CardioML</h1>"
    "<h4 style='text-align:center;'>Explainable ECG-Aware Heart Disease Prediction</h4><hr>",
    unsafe_allow_html=True
)

# ---------------------------------------
# ECG INPUT
# ---------------------------------------
st.subheader("📈 ECG Signal Input")

ecg_file = st.file_uploader(
    "Upload single ECG beat (.csv, 187 values)",
    type=["csv"]
)

ecg_score = None

if ecg_file:
    ecg = pd.read_csv(ecg_file, header=None).values.flatten()
    if len(ecg) != 187:
        st.error("ECG must have exactly 187 values.")
    else:
        ecg_score = get_ecg_risk_score(ecg)
        st.success(f"ECG Signal Risk Score: {ecg_score:.3f}")

# ---------------------------------------
# CLINICAL INPUT
# ---------------------------------------
st.markdown("---")
st.subheader("🧑‍⚕️ Clinical Information")

with st.form("clinical_form"):
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [3, 6, 7])

    submit = st.form_submit_button("🔍 Predict Risk")

# ---------------------------------------
# PREDICTION + SHAP
# ---------------------------------------
if submit and ecg_score is not None:

    input_df = pd.DataFrame([{
        "age": age,
        "sex": 1.0 if sex == "Male" else 0.0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prob = fusion_model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prob >= 0.5:
        st.error(f"⚠️ High Risk (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {prob:.2f})")

    # ---------------------------------------
    # SHAP EXPLANATION
    # ---------------------------------------
    st.markdown("### 🔍 Why this prediction? (Explainable AI)")

    X_trans = shap_preprocessor.transform(input_df)
    shap_vals = shap_explainer.shap_values(X_trans)

    shap_values_patient = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    shap_df = pd.DataFrame({
        "Feature": shap_features,
        "Impact": shap_values_patient
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.dataframe(shap_df.head(6), use_container_width=True)

    st.markdown("#### 🧠 Clinical Interpretation")
    for _, row in shap_df.head(3).iterrows():
        direction = "increases" if row["Impact"] > 0 else "reduces"
        st.write(f"- **{row['Feature']}** {direction} the predicted risk.")




