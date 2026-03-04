import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
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

df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)
df = df.replace("?", np.nan).dropna().astype(float)

X = df.drop(columns=["target"])
y = df["target"]

# ---------------------------------------
# PREPROCESSING (SAME AS TRAINING)
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
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------
# MODELS
# ---------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5,
        learning_rate=0.05, eval_metric="logloss",
        random_state=42
    ),
    "Stacked Ensemble (Proposed)": joblib.load(
        "models/fusion_pipeline.pkl"
    )
}

metrics = {
    "Accuracy": [],
    "Recall": [],
    "F1-score": []
}

model_names = []

# ---------------------------------------
# TRAIN / EVALUATE MODELS
# ---------------------------------------
for name, model in models.items():

    if name != "Stacked Ensemble (Proposed)":
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    metrics["Accuracy"].append(
        accuracy_score(y_test, y_pred)
    )
    metrics["Recall"].append(
        recall_score(y_test, y_pred)
    )
    metrics["F1-score"].append(
        f1_score(y_test, y_pred)
    )

    model_names.append(name)

# ---------------------------------------
# BAR CHART PLOT
# ---------------------------------------
x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(8, 5))

plt.bar(x - width, metrics["Accuracy"], width, label="Accuracy")
plt.bar(x, metrics["Recall"], width, label="Recall")
plt.bar(x + width, metrics["F1-score"], width, label="F1-score")

plt.xticks(x, model_names, rotation=10)
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Model Performance Comparison – Cleveland Dataset")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("graphs/model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Model comparison bar chart saved to graphs/model_comparison.png")
