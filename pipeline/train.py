import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ─── Inference Function — Required for Endpoint Serving ────
def model_fn(model_dir):
    """
    Loads the model for inference.
    SageMaker's SKLearn serving container calls this function
    automatically when the endpoint starts.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


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

    # ✅ Required format for metric_definitions regex capture
    print(f"accuracy: {acc:.4f}")

    print(f"✅ Accuracy : {acc:.2f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("✅ Model saved!")