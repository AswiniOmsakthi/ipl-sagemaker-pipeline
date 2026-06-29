import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train",     type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args = parser.parse_args()

    # ─── Load Data ──────────────────────────────────────────
    train_file = os.path.join(args.train, "train.csv")
    print(f"✅ Files : {os.listdir(args.train)}")

    df = pd.read_csv(train_file)
    print(f"✅ Shape : {df.shape}")

    X = df.drop("team1_won", axis=1)
    y = df["team1_won"]

    # ─── Train ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ─── Evaluate ───────────────────────────────────────────
    preds     = model.predict(X_test)
    accuracy  = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall    = recall_score(y_test, preds)
    f1        = f1_score(y_test, preds)

    # ─── Print in metric_definitions regex format ────────────
    # ✅ These lines make metrics appear in Pipeline Graph UI
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1_score: {f1:.4f}")

    print(f"✅ Accuracy  : {accuracy:.4f}")
    print(f"✅ Precision : {precision:.4f}")
    print(f"✅ Recall    : {recall:.4f}")
    print(f"✅ F1        : {f1:.4f}")

    # ─── Save Model ─────────────────────────────────────────
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("✅ Model saved!")