import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

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
y = df["target"]

# ---------------------------------------
# LOAD TRAINED FUSION PIPELINE
# ---------------------------------------
model = joblib.load("models/fusion_pipeline.pkl")

# ---------------------------------------
# PREDICTION
# ---------------------------------------
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# ---------------------------------------
# CONFUSION MATRIX
# ---------------------------------------
cm = confusion_matrix(y, y_pred)

# ---------------------------------------
# PLOT & SAVE CONFUSION MATRIX
# ---------------------------------------
plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Cleveland Heart Disease Dataset")

plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Confusion matrix saved to graphs/confusion_matrix.png")
print("Confusion Matrix:")
print(cm)
