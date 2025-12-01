import os
import sys
import json
import pickle
import argparse

import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(".."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp from GitHub Actions",
    )
    args = parser.parse_args()

    timestamp = args.timestamp

    # ----------------- 1) Load model: model_<timestamp>_dt_model.joblib -----------------
    try:
        model_version = f"model_{timestamp}_dt_model"
        model_path = f"{model_version}.joblib"
        model = joblib.load(model_path)
        print(f"[INFO] Loaded model from: {model_path}")
    except Exception as e:
        raise ValueError(f"Failed to load the latest model: {e}")

    # ----------------- 2) Load or create Breast Cancer dataset -----------------
    data_dir = "data"
    X_path = os.path.join(data_dir, "data.pickle")
    y_path = os.path.join(data_dir, "target.pickle")

    if os.path.exists(X_path) and os.path.exists(y_path):
        # Use the same data saved during training
        with open(X_path, "rb") as f:
            X = pickle.load(f)
        with open(y_path, "rb") as f:
            y = pickle.load(f)
        print("[INFO] Loaded X, y from existing pickle files.")
    else:
        # Fallback: load directly from sklearn and save for consistency
        print("[INFO] Pickle files not found. Loading Breast Cancer dataset...")
        bc = load_breast_cancer()
        X = bc.data
        y = bc.target

        os.makedirs(data_dir, exist_ok=True)
        with open(X_path, "wb") as f:
            pickle.dump(X, f)
        with open(y_path, "wb") as f:
            pickle.dump(y, f)

    print(f"[INFO] Data shape: X={X.shape}, y={y.shape}")

    # ----------------- 3) Train/test split (same as in train_model.py) -----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # ----------------- 4) Evaluate model -----------------
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "F1_Score": f1,
        "Accuracy": acc,
        "n_samples_test": int(X_test.shape[0]),
    }

    print(f"[INFO] Evaluation metrics: {metrics}")

    # ----------------- 5) Save metrics JSON in CURRENT DIRECTORY -----------------
    # This matches your existing YAML, which does:
    #   metrics_filename="${timestamp}_metrics.json"
    #   mv $metrics_filename Labs/Github_Labs/Lab2/metrics/$metrics_filename

    # Ensure the destination directory under repo root exists (for the mv in YAML)
    workspace = os.getenv("GITHUB_WORKSPACE", os.getcwd())
    lab2_metrics_dir = os.path.join(workspace, "Labs", "Github_Labs", "Lab2", "metrics")
    os.makedirs(lab2_metrics_dir, exist_ok=True)
    print(f"[INFO] Ensured metrics destination dir exists at: {lab2_metrics_dir}")

    # Save the metrics file in the current working directory
    metrics_filename = f"{timestamp}_metrics.json"
    with open(metrics_filename, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"[INFO] Saved metrics to: {metrics_filename}")
