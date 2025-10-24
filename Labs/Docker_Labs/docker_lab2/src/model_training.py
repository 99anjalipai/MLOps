import os, json, time, random, glob
import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

SEED = 42
random.seed(SEED); np.random.seed(SEED)

MODELS_DIR = os.getenv("MODELS_DIR", "models")
METRICS_DIR = os.getenv("METRICS_DIR", "metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def fit_one(Xtr, ytr, Xval, yval, use_pca=False):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    clf = LogisticRegression(max_iter=1000, multi_class="auto", random_state=SEED)
    steps.append(("clf", clf))
    pipe = Pipeline(steps)
    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xval)
    acc = float(accuracy_score(yval, ypred))
    macro_f1 = float(f1_score(yval, ypred, average="macro"))
    cm = confusion_matrix(yval, ypred).tolist()

    pca_components = None
    if use_pca and "pca" in dict(steps):
        pca_components = int(pipe.named_steps["pca"].n_components_)

    return pipe, {
        "val_acc": acc,
        "val_macro_f1": macro_f1,
        "confusion_matrix": cm,
        "pca_components": pca_components
    }

if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    settings = ["baseline", "pca95"]
    per_fold_metrics = {s: [] for s in settings}

    best = {
        "macro_f1": -1.0,
        "acc": -1.0,
        "setting": None,
        "fold": None,
        "model": None
    }

    for fold_id, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        Xtr_raw, Xval_raw = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        #baseline model
        model_a, rep_a = fit_one(Xtr_raw, ytr, Xval_raw, yval, use_pca=False)
        rep_a_out = {**rep_a, "fold": fold_id, "setting": "baseline"}
        per_fold_metrics["baseline"].append(rep_a_out)
        with open(os.path.join(METRICS_DIR, f"cv_baseline_fold_{fold_id}.json"), "w") as f:
            json.dump(rep_a_out, f, indent=2)

        if (rep_a["val_macro_f1"] > best["macro_f1"]) or (
            np.isclose(rep_a["val_macro_f1"], best["macro_f1"]) and rep_a["val_acc"] > best["acc"]
        ):
            best.update({"macro_f1": rep_a["val_macro_f1"], "acc": rep_a["val_acc"],
                         "setting": "baseline", "fold": fold_id, "model": model_a})

        #PCA (retain 95% variance)
        model_b, rep_b = fit_one(Xtr_raw, ytr, Xval_raw, yval, use_pca=True)
        rep_b_out = {**rep_b, "fold": fold_id, "setting": "pca95"}
        per_fold_metrics["pca95"].append(rep_b_out)
        with open(os.path.join(METRICS_DIR, f"cv_pca95_fold_{fold_id}.json"), "w") as f:
            json.dump(rep_b_out, f, indent=2)

        if (rep_b["val_macro_f1"] > best["macro_f1"]) or (
            np.isclose(rep_b["val_macro_f1"], best["macro_f1"]) and rep_b["val_acc"] > best["acc"]
        ):
            best.update({"macro_f1": rep_b["val_macro_f1"], "acc": rep_b["val_acc"],
                         "setting": "pca95", "fold": fold_id, "model": model_b})

    #Summaries
    def summarize(rs):
        accs  = [r["val_acc"] for r in rs]
        f1s   = [r["val_macro_f1"] for r in rs]
        return {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        }

    summary = {
        "cv_n_splits": 5,
        "baseline": summarize(per_fold_metrics["baseline"]),
        "pca95": summarize(per_fold_metrics["pca95"]),
        "best_model": {
            "setting": best["setting"], "fold": best["fold"],
            "val_acc": best["acc"], "val_macro_f1": best["macro_f1"]
        }
    }
    with open(os.path.join(METRICS_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    #Save best model
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"iris_cv_best_{best['setting']}_fold{best['fold']}_{ts}.joblib")
    joblib.dump(best["model"], model_path)

    with open(os.path.join(METRICS_DIR, "labels.json"), "w") as f:
        json.dump(iris.target_names.tolist(), f)

    print("Saved best model to:", model_path)
    print("CV macro-F1 (baseline) mean±std:",
          round(summary["baseline"]["macro_f1_mean"], 4), "±", round(summary["baseline"]["macro_f1_std"], 4))
    print("CV macro-F1 (pca95) mean±std:",
          round(summary["pca95"]["macro_f1_mean"], 4), "±", round(summary["pca95"]["macro_f1_std"], 4))
