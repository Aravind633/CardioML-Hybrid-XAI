import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
# Replace with your actual file name
df = pd.read_csv('data/clinical_data.csv') 

# 2. Preprocessing
# Convert categorical text columns to numbers (e.g., Sex: M/F -> 0/1)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate Features (X) and Target (y)
# Assuming 'target' or 'HeartDisease' is your label column. CHANGE THIS to match your dataset.
X = df.drop(columns=['Heart Attack Risk']) 
y = df['Heart Attack Risk']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [cite_start]3. Model Building (The Ensemble) [cite: 69]
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Voting Classifier (Hybrid)
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft' # Soft voting enables probability output for Risk Scores
)

# 4. Training
print("Training Ensemble Model...")
ensemble_model.fit(X_train, y_train)

# 5. Evaluation
y_pred = ensemble_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 6. Save the Model
with open('models/clinical_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)
print("Clinical model saved to 'models/clinical_ensemble.pkl'")