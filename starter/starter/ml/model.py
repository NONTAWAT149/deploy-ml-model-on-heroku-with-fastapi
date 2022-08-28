from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pytest
from .data import clean_data


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
    return model.predict(X)


def model_performance(data, model):
    performance = {}
    for cat_feature in data['salary'].unique():
        selected_data = data[data['salary'] == cat_feature]
        preds = inference(model, selected_data)
        precision, recall, fbeta = compute_model_metrics(selected_data['salary'], preds)
        performance[cat_feature] = {'precision': precision,
                                    'recall': recall,
                                    'fbeta' : fbeta}
    return performance


@pytest.fixture
def input_data():
    df = pd.read_csv('./data/census.csv')
    df = clean_data(df)
    return df

def test_na_data(input_data):
    assert input_data.shape == input_data.dropna().shape, "need to remove na data"

def test_data_num(input_data):
    assert len(input_data) > 0, "data is emptly"

def test_age_range(input_data):
    assert input_data['age'].min() > 0, "age cannot be negative or zero"
