# Put the code for your API here.
from fastapi import FastAPI
from schema import ModelInput
from ml.model import inference
from ml.data import clean_data


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ML model is ready to use"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/prediction/{data}")
async def model_inference(data: ModelInput):
    data = clean_data(data)
    data = data[['age',
                'workclass',
                'fnlgt',
                'education',
                'education-num',
                'marital-status',
                'occupation',
                'relationship',
                'race',
                'sex',
                'capital-gain',
                'capital-loss',
                'hours-per-week',
                'native-country']]
    return {"prediction": inference(data)}