# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference
from starter.ml.data import process_data
import pandas as pd
import joblib
import os
from fastapi.encoders import jsonable_encoder


print('current directory', os.getcwd())

#GitHub Action Testing
model = joblib.load("./model/model.joblib")
encoder = joblib.load("./model/encoder.joblib")
lb = joblib.load("./model/lb.joblib")

#Local Testing
#model = joblib.load("./model/model.joblib")
#encoder = joblib.load("./model/encoder.joblib")
#lb = joblib.load("./model/lb.joblib")

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

class dataInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {'age': 76,
                 'workclass': 'Private',
                 'fnlgt': 124191,
                 'education': 'Masters',
                 'education-num': 14,
                 'marital-status': 'Married-civ-spouse',
                 'occupation': 'Exec-managerial',
                 'relationship': 'Husband',
                 'race': 'White',
                 'sex': 'Male',
                 'capital-gain': 0,
                 'capital-loss': 0,
                 'hours-per-week': 40,
                 'native-country': 'United-States'}
        }


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ML model is ready for prediction"}

# This allows sending of data (dataInput) via POST to the API.
@app.post("/prediction/")
async def model_inference(data: dataInput):
    data = jsonable_encoder(data)
    data = pd.DataFrame(data=data.values(), index=data.keys()).T
    x_data, _, _, _ = process_data(data,
                                    categorical_features=cat_features,
                                    training=False,
                                    label=None,
                                    encoder=encoder,
                                    lb=lb)
    return {"prediction": inference(model, x_data)}