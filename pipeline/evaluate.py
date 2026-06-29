import pandas as pd
import joblib
import json
import os
import tarfile
import boto3
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─── Install sagemaker inside container ─────────────────────
try:
    subprocess.check_call(["pip", "install", "sagemaker", "--quiet"])
    print("✅ sagemaker installed!")
except Exception as e:
    print(f"⚠️ sagemaker install failed: {str(e)}")

# ─── Paths ──────────────────────────────────────────────────
model_path = "/opt/ml/processing/model/model.tar.gz"
test_path  = "/opt/ml/processing/test/test.csv"
output_dir = "/opt/ml/processing/evaluation"

# ─── Extract Model ──────────────────────────────────────────
with tarfile.open(model_path) as tar:
    tar.extractall("/opt/ml/processing/model/")

# ─── Load Model ─────────────────────────────────────────────
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

# ─── Print Metrics ──────────────────────────────────────────
print(f"✅ Accuracy  : {accuracy:.4f}")
print(f"✅ Precision : {precision:.4f}")
print(f"✅ Recall    : {recall:.4f}")
print(f"✅ F1        : {f1:.4f}")

# ─── Print in SageMaker Metrics Format ──────────────────────
print(f"[SageMaker Metrics] #quality_metric# accuracy={accuracy:.4f};")
print(f"[SageMaker Metrics] #quality_metric# precision={precision:.4f};")
print(f"[SageMaker Metrics] #quality_metric# recall={recall:.4f};")
print(f"[SageMaker Metrics] #quality_metric# f1_score={f1:.4f};")

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

# ─── Log Metrics to SageMaker Experiments ───────────────────
try:
    import sagemaker
    from sagemaker.experiments.run import Run

    boto_session      = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    with Run(
        experiment_name   = "ipl-match-prediction",
        run_name          = "evaluation-run",
        sagemaker_session = sagemaker_session
    ) as run:
        run.log_metric(name="accuracy",  value=round(accuracy,  4), step=1)
        run.log_metric(name="precision", value=round(precision, 4), step=1)
        run.log_metric(name="recall",    value=round(recall,    4), step=1)
        run.log_metric(name="f1_score",  value=round(f1,        4), step=1)
        run.log_parameter("model_type",   "RandomForestClassifier")
        run.log_parameter("n_estimators", "100")
        run.log_parameter("test_size",    "0.2")
        run.log_parameter("dataset",      "IPL 2008-2020")
        run.log_parameter("threshold",    "0.75")

    print("✅ Metrics logged to SageMaker Experiments!")

except Exception as e:
    print(f"⚠️ Experiment logging skipped: {str(e)}")

print(f"✅ Full Report : {json.dumps(report, indent=2)}")