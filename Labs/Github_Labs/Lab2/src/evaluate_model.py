import os
import sys
import json
import pickle
import argparse

import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(".."))


if __name__ == "__main__":
    # ----------------- 1) Parse timestamp argument -----------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp from GitHub Actions to locate the correct model",
    )
    args = parser.parse_args()
    timestamp = args.timestamp

    # ----------------- 2) Load the trained model -----------------
    model_filename = f"model_{timestamp}_rf_pipeline.joblib"
    model_path = os.path.join("models", model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Expected model file not found at: {model_path}"
        )

    try:
        model = joblib.load(model_path)
        print(f"[INFO] Loaded model from: {model_path}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # ----------------- 3) Load dataset from pickle -----------------
    data_path = os.path.join("data", "data.pickle")
    target_path = os.path.join("data", "target.pickle")

    if not (os.path.exists(data_path) and os.path.exists(target_path)):
        raise FileNotFoundError(
            "data/data.pickle and/or data/target.pickle not found. "
            "Make sure train_model.py has been run successfully."
        )

    with open(data_path, "rb") as f_data:
        X = pickle.load(f_data)
    with open(target_path, "rb") as f_target:
        y = pickle.load(f_target)

    print(f"[INFO] Loaded X, y from pickle. X={X.shape}, y={y.shape}")

    # ----------------- 4) Train/test split (same as in train_model.py) -----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # ----------------- 5) Evaluate model on the test set -----------------
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "F1_Score": f1,
        "Accuracy": acc,
        "n_samples_test": int(X_test.shape[0]),
    }

    print(f"[INFO] Evaluation metrics: {metrics}")


    metrics_filename = f"{timestamp}_metrics.json"
    with open(metrics_filename, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"[INFO] Saved metrics to: {metrics_filename}")
