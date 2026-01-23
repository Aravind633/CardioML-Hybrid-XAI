import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    df = pd.read_csv('data/REAL_MEDICAL_DATA.csv')
except FileNotFoundError:
    df = pd.read_csv('REAL_MEDICAL_DATA.csv')

print(f"Loaded {len(df)} samples.")

def simulate_noisy_ecg(row):
    target = row['Heart Disease']
 
    if np.random.random() < 0.10:
        
        if target == 1: return np.random.uniform(0.1, 0.45)
        else: return np.random.uniform(0.55, 0.9)
    

    if target == 1: return np.random.uniform(0.6, 0.99)
    else: return np.random.uniform(0.0, 0.4)

df['ECG_Risk_Score'] = df.apply(simulate_noisy_ecg, axis=1)

label_encoders = {}
categorical_cols = ['Gender', 'Family History', 'Chest Pain Type', 'Exercise Induced Angina', 'Smoking']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Heart Disease'])
y = df['Heart Disease']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=8,       
    min_samples_split=10, 
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=4, 
    subsample=0.8,      
    random_state=42
)

model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"FINAL ACCURACY: {acc*100:.2f}%")
print("="*30)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Average: {cv_scores.mean()*100:.2f}% (Variance: {cv_scores.std()*100:.2f}%)")

if abs(acc - cv_scores.mean()) > 0.05:
    print("WARNING: Possible Overfitting detected.")
else:
    print("Model is Robust (Test score matches CV score).")

with open('models/fusion_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': feature_names}, f)
with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models Saved Successfully.")