# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.model import train_model, model_performance
from ml.data import process_data, clean_data
from joblib import dump

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
X_train, y_train, encoder, lb = process_data(
                                    train,
                                    categorical_features=cat_features,
                                    label="salary",
                                    training=True)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
                                    test,
                                    categorical_features=cat_features,
                                    label="salary",
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)

# Train and save a model.
trained_model = train_model(X_train, y_train)
dump(trained_model, "../model/model.joblib")
dump(encoder, "../model/encoder.joblib")
dump(lb, "../model/lb.joblib")

# Model Evaluation of testing data
performance = model_performance(test,
                                trained_model,
                                encoder,
                                lb,
                                cat_features,
                                'education')
print('model_performance: ', performance)

# open file for writing
f = open("slice_output.txt","w")
# write file
f.write( str(performance) )
# close file
f.close()