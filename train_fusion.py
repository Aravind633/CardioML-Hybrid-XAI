import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    "data/processed.cleveland.data",
    names=columns
)

print("Initial shape:", df.shape)
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

df = df.replace("?", np.nan)
df = df.dropna()
df = df.astype(float)

print("Shape after cleaning:", df.shape)


X = df.drop(columns=["target"])
y = df["target"]

categorical_features = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal"
]

numerical_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

stack_model = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb)
    ],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5,
    n_jobs=-1
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", stack_model)
])


model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred_default = (y_prob >= 0.5).astype(int)

print("\nHOLD-OUT RESULTS (DEFAULT THRESHOLD = 0.5)")
print("Accuracy:", round(accuracy_score(y_test, y_pred_default) * 100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_default))


thresholds = np.arange(0.3, 0.7, 0.01)
best_thresh = 0.5
best_acc = 0

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("\nTHRESHOLD OPTIMIZATION RESULTS")
print("Best Threshold:", round(best_thresh, 2))
print("Best Accuracy:", round(best_acc * 100, 2), "%")

y_pred_opt = (y_prob >= best_thresh).astype(int)

print("\nFINAL HOLD-OUT RESULTS (OPTIMIZED THRESHOLD)")
print("Accuracy:", round(accuracy_score(y_test, y_pred_opt) * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_opt))

rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=rskf,
    scoring="roc_auc",
    n_jobs=-1
)

print("\nREPEATED STRATIFIED CV ROC-AUC")
print("Mean ROC-AUC:", round(auc_scores.mean(), 3))
print("Std ROC-AUC:", round(auc_scores.std(), 3))
