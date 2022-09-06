# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.model import inference_dev
from ml.data import process_data, clean_data
from sklearn.metrics import classification_report
import joblib

# Add code to load in the data.
df = pd.read_csv('../data/census.csv')
data = clean_data(df)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
encoder = joblib.load("../model/encoder.joblib")
lb = joblib.load("../model/lb.joblib")

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
                                    test,
                                    categorical_features=cat_features,
                                    label="salary",
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)

trained_model = joblib.load("../model/model.joblib")
y_prediction = inference_dev(trained_model, X_test)

#get model performance report
report = classification_report(y_test, y_prediction, output_dict=True)
print(report)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv('model_performance_report.csv')
