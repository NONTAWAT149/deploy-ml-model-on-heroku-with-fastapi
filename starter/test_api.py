from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200, "response not successful"
    assert response.json() == {"message": "ML model is ready for prediction"}, "wrong message gotten"

def test_post_inference_one():
    input_data = {'age': 76,
                 'workclass': 'Private',
                 'fnlgt': 124191,
                 'education': 'Masters',
                 'education_num': 14,
                 'marital_status': 'Married-civ-spouse',
                 'occupation': 'Exec-managerial',
                 'relationship': 'Husband',
                 'race': 'White',
                 'sex': 'Male',
                 'capital_gain': 0,
                 'capital_loss': 0,
                 'hours_per_week': 40,
                 'native_country': 'United-States'}

    response_post = client.post("/prediction/", json=input_data)
    assert response_post.status_code == 200, "response not successful with {}".format(response_post.json())
    assert response_post.json() == {"prediction":'>50K'}, "got wrong prediction, expect '>50K'"


def test_post_inference_two():
    input_data = {'age': 22,
                 'workclass': 'Private',
                 'fnlgt': 201490,
                 'education': 'HS-grad',
                 'education_num': 9,
                 'marital_status': 'Never-married',
                 'occupation': 'Adm-clerical',
                 'relationship': 'Own-child',
                 'race': 'White',
                 'sex': 'Male',
                 'capital_gain': 0,
                 'capital_loss': 0,
                 'hours_per_week': 20,
                 'native_country': 'United-States'}

    response_post = client.post("/prediction/", json=input_data)
    assert response_post.status_code == 200, "response not successful with {}".format(response_post.json())
    assert response_post.json() == {"prediction":'<=50K'}, "got wrong prediction, expect '<=50K'"


