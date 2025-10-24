from xgboost import XGBClassifier
import joblib
from data import load_data, split_data
import numpy as np

def fit_model(X_train, y_train):
    """
    Train a XGBoost Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    n_classes = int(len(np.unique(y_train)))

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob", 
        num_class=n_classes,
        random_state=12,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "../model/iris_xgboost_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
