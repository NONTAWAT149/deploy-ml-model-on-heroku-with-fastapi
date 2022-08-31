from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from .data import clean_data, process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    prediction_value = model.predict(X)[0]
    if prediction_value == 1:
        prediction_message = '>50K'
    if prediction_value == 0:
        prediction_message = '<=50K'
    return prediction_message


def inference_dev(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def model_performance(data, model, encoder, lb, cat_features):
    performance = {}
    for cat_feature in data['salary'].unique():
        selected_data = data[data['salary'] == cat_feature]
        X_test, y_test, _, _ = process_data(selected_data,
                                 categorical_features=cat_features,
                                 label="salary",
                                 training=False,
                                 encoder=encoder,
                                 lb=lb)
        preds = inference_dev(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        performance[cat_feature] = {'precision': precision,
                                    'recall': recall,
                                    'fbeta' : fbeta}
    return performance

