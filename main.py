from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# App instance
app = FastAPI()

# Input schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                      features.AveBedrms, features.Population,
                      features.AveOccup, features.Latitude, features.Longitude]])
    prediction = model.predict(data)[0]
    return {"predicted_price": round(prediction, 2)}
