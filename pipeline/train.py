import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ─── Inference Functions — Required for Endpoint Serving ───
def model_fn(model_dir):
    """Loads the model for inference."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):
    """Parses incoming CSV request and ensures it's 2D for sklearn."""
    import io

    if request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        return df.values  # ✅ 2D numpy array
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Runs prediction on properly shaped 2D input."""
    return model.predict(input_data)


def output_fn(prediction, content_type):
    """Formats prediction as plain CSV text."""
    import io
    import numpy as np

    output = io.StringIO()
    for p in np.atleast_1d(prediction):
        output.write(f"{p}\n")
    return output.getvalue()


# ─── Training Entry Point ───────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args = parser.parse_args()

    train_file = os.path.join(args.train, "train.csv")
    print(f"✅ Files : {os.listdir(args.train)}")

    df = pd.read_csv(train_file)
    print(f"✅ Shape : {df.shape}")

    X = df.drop("team1_won", axis=1)
    y = df["team1_won"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f"accuracy: {acc:.4f}")
    print(f"✅ Accuracy : {acc:.2f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("✅ Model saved!")