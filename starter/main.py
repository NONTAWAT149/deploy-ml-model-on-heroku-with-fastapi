# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference
import pandas as pd

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


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ML model is ready for prediction"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/prediction/")
async def model_inference(data: dataInput):
    data = pd.DataFrame(data.dict(), index=[0])
    return {"prediction": inference(data)}