import joblib
from pathlib import Path

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = Path(__file__).resolve().parent.parent / "model" / "iris_xgboost_model.pkl"
    model = joblib.load(model)
    y_pred = model.predict(X)
    return y_pred
