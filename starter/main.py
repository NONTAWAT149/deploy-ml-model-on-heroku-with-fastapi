# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from ml.model import inference
import pandas as pd

class dataInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education-num: int
    marital-status: str
    occupation: ste
    relationship: str
    race: str
    sex: str
    capital-gain: int
    capital-loss: int
    hours-per-week: int
    native-country: str


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ML model is ready for prediction"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/prediction/{data}")
async def model_inference(data: dataInput):
    data = pd.DataFrame(list(data.values()), index=list(data.keys())).T
    return {"prediction": inference(data)}