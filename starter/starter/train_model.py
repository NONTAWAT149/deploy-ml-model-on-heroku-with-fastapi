# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from model import train_model
from joblib import dump

# Add code to load in the data.
df = pd.read_csv('./data/census.csv')

# clean data by removing space
# clean column name
column_list = list(df.columns)

column_rename = {}
for original_column in list(df.columns):
    column_rename[original_column] = original_column.strip()

df.rename(columns=column_rename, inplace = True)

# clean data content
for column in list(df.columns):
    if df[column].dtypes == 'object':
        df[column] = df[column].apply(lambda x: x.strip())

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

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
dump(trained_model, "./model/model.joblib")