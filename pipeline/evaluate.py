import pandas as pd
import joblib
import json
import os
import tarfile
import boto3
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─── Paths ──────────────────────────────────────────────────
model_path = "/opt/ml/processing/model/model.tar.gz"
test_path  = "/opt/ml/processing/test/test.csv"
output_dir = "/opt/ml/processing/evaluation"

# ─── Extract & Load Model ───────────────────────────────────
with tarfile.open(model_path) as tar:
    tar.extractall("/opt/ml/processing/model/")

model = joblib.load("/opt/ml/processing/model/model.joblib")

# ─── Load Test Data ─────────────────────────────────────────
df    = pd.read_csv(test_path)
X     = df.drop("team1_won", axis=1)
y     = df["team1_won"]

# ─── Predict ────────────────────────────────────────────────
preds     = model.predict(X)
accuracy  = accuracy_score(y, preds)
precision = precision_score(y, preds)
recall    = recall_score(y, preds)
f1        = f1_score(y, preds)

print(f"✅ Accuracy  : {accuracy:.4f}")
print(f"✅ Precision : {precision:.4f}")
print(f"✅ Recall    : {recall:.4f}")
print(f"✅ F1        : {f1:.4f}")

# ─── Save Evaluation Report ─────────────────────────────────
os.makedirs(output_dir, exist_ok=True)

report = {
    "metrics": {
        "accuracy"  : {"value": round(accuracy,  4)},
        "precision" : {"value": round(precision, 4)},
        "recall"    : {"value": round(recall,    4)},
        "f1_score"  : {"value": round(f1,        4)}
    }
}

with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
    json.dump(report, f, indent=2)

print("✅ Evaluation report saved!")
print(f"✅ Full Report : {json.dumps(report, indent=2)}")