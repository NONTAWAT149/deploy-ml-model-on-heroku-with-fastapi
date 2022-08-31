import pandas as pd
import pytest
from starter.ml.data import clean_data


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