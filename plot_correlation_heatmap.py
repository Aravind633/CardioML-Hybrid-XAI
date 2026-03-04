import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# ---------------------------------------
# CORRELATION MATRIX
# ---------------------------------------
corr_matrix = df.corr(method="pearson")

# ---------------------------------------
# PLOT HEATMAP
# ---------------------------------------
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar=True,
    linewidths=0.5
)

plt.title("Correlation Heatmap – Cleveland Heart Disease Dataset")
plt.tight_layout()

# ---------------------------------------
# SAVE FIGURE
# ---------------------------------------
plt.savefig("graphs/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Correlation heatmap saved to graphs/correlation_heatmap.png")
