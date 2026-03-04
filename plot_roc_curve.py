import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

# ---------------------------------------
# LOAD CLEVELAND DATASET
# ---------------------------------------
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv("data/processed.cleveland.data", names=columns)

# Binary target: 0 = no disease, 1 = disease
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# Clean dataset
df = df.replace("?", np.nan)
df = df.dropna()
df = df.astype(float)

X = df.drop(columns=["target"])
y = df["target"]

# ---------------------------------------
# LOAD TRAINED FUSION PIPELINE
# ---------------------------------------
model = joblib.load("models/fusion_pipeline.pkl")

# ---------------------------------------
# PREDICT PROBABILITIES
# ---------------------------------------
y_prob = model.predict_proba(X)[:, 1]

# ---------------------------------------
# COMPUTE ROC CURVE
# ---------------------------------------
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)

# ---------------------------------------
# PLOT & SAVE ROC CURVE
# ---------------------------------------
plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Fusion Model (AUC = {auc_score:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random Guess"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Cleveland Heart Disease Dataset")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()

# Save figure
plt.savefig("graphs/roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ ROC curve saved to graphs/roc_curve.png")
print(f"📈 ROC-AUC Score: {auc_score:.4f}")
