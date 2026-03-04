

# import pandas as pd
# import numpy as np
# import joblib
# import shap

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score
# from xgboost import XGBClassifier

# # ---------------------------------------
# # LOAD DATA
# # ---------------------------------------
# columns = [
#     "age", "sex", "cp", "trestbps", "chol", "fbs",
#     "restecg", "thalach", "exang", "oldpeak",
#     "slope", "ca", "thal", "target"
# ]

# df = pd.read_csv("data/processed.cleveland.data", names=columns)
# df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)
# df = df.replace("?", np.nan).dropna().astype(float)

# X = df.drop(columns=["target"])
# y = df["target"]

# # ---------------------------------------
# # PREPROCESSING
# # ---------------------------------------
# categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
# numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#     ("num", "passthrough", numerical_features)
# ])

# # ---------------------------------------
# # TRAIN / TEST SPLIT
# # ---------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # ---------------------------------------
# # MODELS
# # ---------------------------------------
# rf = RandomForestClassifier(
#     n_estimators=500, max_depth=12,
#     min_samples_leaf=3, random_state=42, n_jobs=-1
# )

# xgb = XGBClassifier(
#     n_estimators=400, max_depth=5, learning_rate=0.05,
#     subsample=0.9, colsample_bytree=0.9,
#     eval_metric="logloss", random_state=42
# )

# stack = StackingClassifier(
#     estimators=[("rf", rf), ("xgb", xgb)],
#     final_estimator=LogisticRegression(max_iter=2000),
#     cv=5, n_jobs=-1
# )

# # ---------------------------------------
# # FULL PIPELINE
# # ---------------------------------------
# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier", stack)
# ])

# pipeline.fit(X_train, y_train)

# # ---------------------------------------
# # EVALUATION
# # ---------------------------------------
# y_prob = pipeline.predict_proba(X_test)[:, 1]
# print("Accuracy:", accuracy_score(y_test, (y_prob >= 0.5).astype(int)))
# print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# # ---------------------------------------
# # SAVE PIPELINE
# # ---------------------------------------
# joblib.dump(pipeline, "models/fusion_pipeline.pkl")
# print("✅ Fusion pipeline saved")

# # ---------------------------------------
# # SHAP EXPLAINER (STAGE 3)
# # ---------------------------------------
# preprocessor_fitted = pipeline.named_steps["preprocessor"]
# stack_model = pipeline.named_steps["classifier"]
# xgb_model = stack_model.named_estimators_["xgb"]

# X_train_transformed = preprocessor_fitted.transform(X_train)
# explainer = shap.TreeExplainer(xgb_model)

# cat_feature_names = (
#     preprocessor_fitted
#     .named_transformers_["cat"]
#     .get_feature_names_out(categorical_features)
# )

# feature_names = list(cat_feature_names) + numerical_features

# joblib.dump({
#     "explainer": explainer,
#     "preprocessor": preprocessor_fitted,
#     "feature_names": feature_names
# }, "models/shap_explainer.pkl")

# print("✅ SHAP explainer saved")






import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv("data/processed.cleveland.data", names=columns)

# Binary target: 0 = no disease, 1 = disease
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# Clean data
df = df.replace("?", np.nan)
df = df.dropna()
df = df.astype(float)

X = df.drop(columns=["target"])
y = df["target"]

# ---------------------------------------
# PREPROCESSING
# ---------------------------------------
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numerical_features)
])

# ---------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ---------------------------------------
# MODELS
# ---------------------------------------
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

stack = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5,
    n_jobs=-1
)

# ---------------------------------------
# FULL PIPELINE
# ---------------------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", stack)
])

pipeline.fit(X_train, y_train)

# ---------------------------------------
# EVALUATION (COMPLETE METRICS)
# ---------------------------------------
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n📊 MODEL EVALUATION RESULTS (HOLD-OUT TEST SET)\n")

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

print("📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------
# SAVE PIPELINE
# ---------------------------------------
joblib.dump(pipeline, "models/fusion_pipeline.pkl")
print("\n✅ Fusion pipeline saved to models/fusion_pipeline.pkl")

# ---------------------------------------
# SHAP EXPLAINER (STAGE 3 – XAI)
# ---------------------------------------
preprocessor_fitted = pipeline.named_steps["preprocessor"]
stack_model = pipeline.named_steps["classifier"]
xgb_model = stack_model.named_estimators_["xgb"]

X_train_transformed = preprocessor_fitted.transform(X_train)
explainer = shap.TreeExplainer(xgb_model)

cat_feature_names = (
    preprocessor_fitted
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_features)
)

feature_names = list(cat_feature_names) + numerical_features

joblib.dump(
    {
        "explainer": explainer,
        "preprocessor": preprocessor_fitted,
        "feature_names": feature_names
    },
    "models/shap_explainer.pkl"
)

print("✅ SHAP explainer saved to models/shap_explainer.pkl")
