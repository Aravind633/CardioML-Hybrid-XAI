import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------------------
# LOAD CLEVELAND DATASET
# ---------------------------------------
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv("data/processed.cleveland.data", names=columns)

# Binary target
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# Clean dataset
df = df.replace("?", np.nan)
df = df.dropna()
df = df.astype(float)

X = df.drop(columns=["target"])

# ---------------------------------------
# LOAD SHAP ARTIFACTS
# ---------------------------------------
shap_obj = joblib.load("models/shap_explainer.pkl")

explainer = shap_obj["explainer"]
preprocessor = shap_obj["preprocessor"]
feature_names = shap_obj["feature_names"]

# ---------------------------------------
# TRANSFORM DATA (IMPORTANT)
# ---------------------------------------
X_transformed = preprocessor.transform(X)

# ---------------------------------------
# COMPUTE SHAP VALUES
# ---------------------------------------
shap_values = explainer.shap_values(X_transformed)

# For binary classification → class 1
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# ---------------------------------------
# SHAP SUMMARY PLOT (GLOBAL IMPORTANCE)
# ---------------------------------------
plt.figure(figsize=(8, 6))

shap.summary_plot(
    shap_vals,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig("graphs/shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ SHAP summary plot saved to graphs/shap_summary_plot.png")
