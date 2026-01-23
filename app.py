import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CardioML Fusion", layout="wide")

@st.cache_resource
def load_system():
    # Load Deep Learning Model
    ecg_model = tf.keras.models.load_model('models/ecg_lstm_model.h5')
    
    # Load ML Artifacts
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
except Exception as e:
    st.error(f"System Load Error: {e}")
    st.stop()


st.title("🫀 CardioML: Multi-Modal Fusion System")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Temporal Signal Analysis (ECG)")
    uploaded_file = st.file_uploader("Upload ECG CSV (1-Lead)", type=["csv"])
    
    
    ecg_risk_score = 0.1 
    
    if uploaded_file:
        try:
            df_ecg = pd.read_csv(uploaded_file, header=None)
            signal = df_ecg.iloc[0, :187].values.astype(float)
            st.line_chart(signal)
            
            # Predict using LSTM
            signal_reshaped = signal.reshape(1, 187, 1)
            probs = ecg_model.predict(signal_reshaped)[0]
            
          
            ecg_risk_score = float(np.sum(probs[1:])) 
            st.metric("Deep Learning Signal Score", f"{ecg_risk_score:.2f}")
            
        except Exception as e:
            st.error(f"Error processing ECG: {e}")

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

if st.button("Analyze Combined Risk", type="primary"):
   
    input_data = {
        'Age': age, 'Gender': gender, 'Family History': family,
        'Resting BP': bp, 'Cholesterol': chol, 'Chest Pain Type': cp_type,
        'Exercise Induced Angina': ex_angina, 'Smoking': smoking,
        'Physical Activity (hr/wk)': activity,
        'ECG_Risk_Score': ecg_risk_score 
    }
    
    df_in = pd.DataFrame([input_data])
    
    
    for col, le in encoders.items():
        if col in df_in.columns:
            val = df_in[col].iloc[0]
          
            df_in[col] = le.transform([val]) if val in le.classes_ else 0
            
    
    df_in = df_in[feature_names]
    X_scaled = scaler.transform(df_in)
    
 
    prob_array = fusion_model.predict_proba(X_scaled)
    prob = float(prob_array[0][1]) 
    
    st.divider()
    
    
    st.markdown(f"### Final Integrated Risk: {prob:.1%}")
    if prob > 0.5:
        st.error("⚠️ HIGH RISK DETECTED")
    else:
        st.success("✅ LOW RISK PROFILE")


    st.subheader("3. XAI: Patient-Specific Explanation")
    
    try:
       
        rf_model = fusion_model.estimators_[0] 
        explainer = shap.TreeExplainer(rf_model)
        

        shap_values = explainer.shap_values(X_scaled)
        
        #
        if isinstance(shap_values, list):
            vals = shap_values[1] 
        else:
            if len(shap_values.shape) == 3: 
                vals = shap_values[:,:,1] 
            else:
                vals = shap_values

        vals = np.array(vals).flatten()
        vals = [float(v) for v in vals] 
        
       
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        readable_map = {
            'ECG_Risk_Score': 'detected ECG irregularities',
            'Age': 'patient age',
            'Resting BP': 'elevated blood pressure',
            'Cholesterol': 'high cholesterol levels',
            'Chest Pain Type': 'reported chest pain symptoms',
            'Smoking': 'smoking history',
            'Family History': 'family history of heart disease',
            'Physical Activity (hr/wk)': 'lack of physical activity',
            'Exercise Induced Angina': 'exercise-induced angina'
        }
        
        # Pair features with their SHAP impact
        feature_impacts = list(zip(feature_names, vals))
        
       
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        
        if prob > 0.5:
            
            drivers = [f for f, v in feature_impacts if v > 0]
            status_text = "Risk is High"
            intro_phrase = "Key contributing factors include"
        else:
            
            drivers = [f for f, v in feature_impacts if v < 0]
            status_text = "Risk is Low"
            intro_phrase = "Protective factors observed include"
            
           
            readable_map['Physical Activity (hr/wk)'] = 'healthy physical activity levels'
            readable_map['Resting BP'] = 'normal blood pressure'
            readable_map['Cholesterol'] = 'controlled cholesterol'

        
        top_factors = drivers[:3] 
        human_names = [readable_map.get(f, f) for f in top_factors]
        
        
        if len(human_names) > 0:
            if len(human_names) == 1:
                factors_str = human_names[0]
            elif len(human_names) == 2:
                factors_str = f"{human_names[0]} and {human_names[1]}"
            else:
                factors_str = ", ".join(human_names[:-1]) + ", and " + human_names[-1]
            
            explanation = f"**{status_text}.** {intro_phrase} **{factors_str}**."
            st.info(f"📝 **AI Clinical Analysis:** {explanation}")
        else:
            st.info("📝 **AI Clinical Analysis:** Risk factors are balanced with no single dominant driver.")

       
        plt.clf()
        
       
        shap.force_plot(
            base_value, 
            np.array(vals), 
            df_in.iloc[0].values, 
            feature_names=feature_names, 
            matplotlib=True, 
            show=False,
            figsize=(20, 3) 
        )
   
        fig = plt.gcf()
        
       
        st.pyplot(fig, bbox_inches='tight', dpi=300)
        
    except Exception as e:
        st.warning(f"XAI Error: {e}")