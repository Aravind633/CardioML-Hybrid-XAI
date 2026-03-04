import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score

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
# PRECISION–RECALL CURVE
# ---------------------------------------
precision, recall, thresholds = precision_recall_curve(y, y_prob)
avg_precision = average_precision_score(y, y_prob)

# ---------------------------------------
# PLOT & SAVE PR CURVE
# ---------------------------------------
plt.figure(figsize=(6, 5))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"Fusion Model (AP = {avg_precision:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Cleveland Heart Disease Dataset")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graphs/precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Precision–Recall curve saved to graphs/precision_recall_curve.png")
print(f"📈 Average Precision (AP): {avg_precision:.4f}")
