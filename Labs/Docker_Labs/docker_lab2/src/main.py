from flask import Flask, request, jsonify, render_template
import numpy as np
import json, os, glob, threading
import joblib

app = Flask(__name__, static_folder='statics') 
lock = threading.Lock()

MODEL_DIR = os.getenv("MODELS_DIR", "models")
METRICS_DIR = os.getenv("METRICS_DIR", "metrics")

DEFAULT_LABELS = ['Setosa', 'Versicolor', 'Virginica']

_model = None
_class_labels = DEFAULT_LABELS

def _latest(path_glob: str):
    files = glob.glob(path_glob)
    return max(files, key=os.path.getmtime) if files else None

def load_artifacts():
    """
    Load latest sklearn Pipeline (.joblib) and labels.
    The pipeline already includes StandardScaler + PCA and the classifier.
    """
    global _model, _class_labels
    with lock:
        model_path = _latest(os.path.join(MODEL_DIR, "iris_cv_best_*.joblib")) \
                     or _latest(os.path.join(MODEL_DIR, "*.joblib"))
        if not model_path:
            raise RuntimeError("No .joblib model found in models/")

        _model = joblib.load(model_path)

        labels_path = _latest(os.path.join(METRICS_DIR, "labels.json"))
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                _class_labels = json.load(f)
        else:
            _class_labels = DEFAULT_LABELS

        print(f"[load_artifacts] model={model_path} labels={labels_path or '(default)'}")

def predict_proba_one(x_vec):
    """
    x_vec: list[float] = [sepal_length, sepal_width, petal_length, petal_width]
    returns: (pred_label:str, probs:list[float])
    """
    if _model is None:
        raise RuntimeError("Model not loaded")
    X = np.asarray(x_vec, dtype=float).reshape(1, -1)  
    probs = _model.predict_proba(X)[0]                 
    idx = int(np.argmax(probs))
    return _class_labels[idx], probs.tolist()

@app.route("/health")
def health():
    ok = (_model is not None)
    return jsonify({"status": "ok" if ok else "not_ready",
                    "model_loaded": ok,
                    "n_classes": len(_class_labels)})

@app.route("/")
def home():
    return "Welcome to the Iris Classifier API!"

@app.route("/reload", methods=["POST"])
def reload_artifacts_route():
    try:
        load_artifacts()
        return jsonify({"status": "reloaded"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            try:
                return render_template('predict.html')
            except Exception:
                return jsonify({
                    "message": "POST JSON or form data to /predict",
                    "schema": {"sepal_length": "float",
                               "sepal_width": "float",
                               "petal_length": "float",
                               "petal_width": "float"}
                }), 200

        # POST
        data = request.get_json(force=True) if request.is_json else request.form
        required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        sepal_length = float(data["sepal_length"])
        sepal_width  = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width  = float(data["petal_width"])

        label, probs = predict_proba_one([sepal_length, sepal_width, petal_length, petal_width])
        return jsonify({
            "predicted_class": label,
            "probabilities": dict(zip(_class_labels, probs))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True, host='0.0.0.0', port=4000)
