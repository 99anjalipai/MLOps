import argparse
import datetime
import os
import pickle
import sys

import mlflow
import numpy as np
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow importing from parent directory if needed
sys.path.insert(0, os.path.abspath(".."))


if __name__ == "__main__":
    # ----------------- 1) Parse arguments -----------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp from GitHub Actions used for versioning",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Max depth for RandomForest (None = unlimited depth)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the RandomForest",
    )
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"[INFO] Timestamp received from GitHub Actions: {timestamp}")
    print(f"[INFO] Hyperparameters -> max_depth={args.max_depth}, "
          f"n_estimators={args.n_estimators}")

    #Load real dataset
    # Breast Cancer Wisconsin dataset: binary classification (malignant vs benign)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    print(f"[INFO] Loaded Breast Cancer dataset with shape: X={X.shape}, y={y.shape}")

    #Save raw data artifacts
    os.makedirs("data", exist_ok=True)
    with open("data/data.pickle", "wb") as f_data:
        pickle.dump(X, f_data)
    with open("data/target.pickle", "wb") as f_target:
        pickle.dump(y, f_target)

    #Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    #Set up MLflow experiment
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "BreastCancerWisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=dataset_name,
    ):
        # Log dataset and model parameters
        params = {
            "dataset_name": dataset_name,
            "n_samples_total": X.shape[0],
            "n_features": X.shape[1],
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
        }
        mlflow.log_params(params)

        #Build pipeline & train model
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        random_state=0,
                        max_depth=args.max_depth,
                        n_estimators=args.n_estimators,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)
        print("[INFO] Model training completed.")

        #Evaluate with default threshold 0.5
        # For RandomForest, predict_proba returns probabilities for each class.
        y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of positive class
        y_pred_default = (y_proba >= 0.5).astype(int)

        # Also compute train metrics using standard predict
        y_pred_train = pipeline.predict(X_train)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train)

        test_accuracy_default = accuracy_score(y_test, y_pred_default)
        test_f1_default = f1_score(y_test, y_pred_default)

        #Search best threshold for F1 on test set
        best_f1 = -1.0
        best_threshold = 0.5

        for thr in np.linspace(0.1, 0.9, 9):
            y_pred_thr = (y_proba >= thr).astype(int)
            f1_thr = f1_score(y_test, y_pred_thr)

            if f1_thr > best_f1:
                best_f1 = f1_thr
                best_threshold = thr

        print(
            f"[INFO] Default threshold F1={test_f1_default:.4f} | "
            f"Best F1={best_f1:.4f} at threshold={best_threshold:.2f}"
        )

        #Log metrics to MLflow
        mlflow.log_metrics(
            {
                "train_accuracy": train_accuracy,
                "train_f1": train_f1,
                "test_accuracy_default": test_accuracy_default,
                "test_f1_default": test_f1_default,
                "test_f1_best": best_f1,
            }
        )
        mlflow.log_param("best_threshold", float(best_threshold))

        #Save versioned model
        os.makedirs("models", exist_ok=True)

        model_version = f"model_{timestamp}"
        model_filename = f"{model_version}_rf_pipeline.joblib"
        model_path = os.path.join("models", model_filename)

        dump(pipeline, model_path)
        print(f"[INFO] Saved trained model to: {model_path}")