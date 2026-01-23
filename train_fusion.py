# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score

# # ==========================================
# # 1. LOAD CLINICAL DATA
# # ==========================================
# df = pd.read_csv('data/REAL_MEDICAL_DATA.csv')

# # ==========================================
# # 2. FEATURE FUSION (Adding Stage 1 Output)
# # ==========================================
# # We simulate the ECG Risk Score for training purposes.
# # Sick people (1) get higher ECG scores (0.6-0.9). Healthy (0) get lower (0.0-0.4).
# print("Fusing ECG Signal Risk Score...")

# def simulate_ecg_score(target):
#     if target == 1:
#         return np.random.uniform(0.5, 0.99) # High risk score
#     else:
#         return np.random.uniform(0.01, 0.45) # Low risk score

# df['ECG_Risk_Score'] = df['Heart Disease'].apply(simulate_ecg_score)

# # ==========================================
# # 3. PREPROCESSING
# # ==========================================
# label_encoders = {}
# categorical_cols = ['Gender', 'Family History', 'Chest Pain Type', 'Exercise Induced Angina', 'Smoking']

# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# X = df.drop(columns=['Heart Disease'])
# y = df['Heart Disease']
# feature_names = X.columns.tolist()

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ==========================================
# # 4. STACKED ENSEMBLE (Stage 2 Model)
# # ==========================================
# rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)

# # Voting Regressor/Classifier
# model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
# model.fit(X_train_scaled, y_train)

# # Evaluate
# acc = accuracy_score(y_test, model.predict(X_test_scaled))
# print(f"Stage 2 Accuracy (Clinical + ECG): {acc*100:.2f}%")

# # ==========================================
# # 5. SAVE
# # ==========================================
# with open('models/encoders.pkl', 'wb') as f:
#     pickle.dump(label_encoders, f)
# with open('models/scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
# with open('models/fusion_model.pkl', 'wb') as f:
#     pickle.dump({'model': model, 'features': feature_names}, f)

# print("✅ Stage 2 Complete: Fusion Model Saved.")






import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. LOAD DATA (Robust Path Checking)
# ==========================================
try:
    df = pd.read_csv('data/REAL_MEDICAL_DATA.csv')
except FileNotFoundError:
    df = pd.read_csv('REAL_MEDICAL_DATA.csv')

print(f"Loaded {len(df)} samples.")

# ==========================================
# 2. SIMULATE ECG (With 10% Noise)
# ==========================================
# This prevents the model from relying 100% on the ECG score.
# It forces the model to look at Clinical Data too.
def simulate_noisy_ecg(row):
    target = row['Heart Disease']
    
    # 10% Chance of Device Error / Human Error
    if np.random.random() < 0.10:
        # Error Case: Sick person gets Low Score, or Healthy gets High
        if target == 1: return np.random.uniform(0.1, 0.45)
        else: return np.random.uniform(0.55, 0.9)
    
    # 90% Case: Correct Correlation
    if target == 1: return np.random.uniform(0.6, 0.99)
    else: return np.random.uniform(0.0, 0.4)

df['ECG_Risk_Score'] = df.apply(simulate_noisy_ecg, axis=1)

# ==========================================
# 3. PREPROCESSING
# ==========================================
label_encoders = {}
categorical_cols = ['Gender', 'Family History', 'Chest Pain Type', 'Exercise Induced Angina', 'Smoking']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Heart Disease'])
y = df['Heart Disease']
feature_names = X.columns.tolist()

# Split Data (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale (Fit on Train, Transform on Test -> NO LEAKAGE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. ROBUST MODEL TRAINING
# ==========================================
# We limit 'max_depth' to prevent memorization (Overfitting)
rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=8,       # Limited depth forces generalization
    min_samples_split=10, 
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=0.05, # Slower learning = better patterns
    max_depth=4, 
    subsample=0.8,      # Uses 80% of data per tree (adds robustness)
    random_state=42
)

# Voting Classifier averages the errors of both models
model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
model.fit(X_train_scaled, y_train)

# ==========================================
# 5. EVALUATION
# ==========================================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"FINAL ACCURACY: {acc*100:.2f}%")
print("="*30)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Double Check: Cross Validation
# This runs the training 5 times on different splits to prove stability
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Average: {cv_scores.mean()*100:.2f}% (Variance: {cv_scores.std()*100:.2f}%)")

if abs(acc - cv_scores.mean()) > 0.05:
    print("WARNING: Possible Overfitting detected.")
else:
    print("Model is Robust (Test score matches CV score).")

# ==========================================
# 6. SAVE
# ==========================================
with open('models/fusion_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': feature_names}, f)
with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models Saved Successfully.")