from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import start_http_server, Counter

app = FastAPI()
MODEL_PATH = "../models/latest_model.pkl"

# Prometheus metrics
PREDICTION_COUNTER = Counter('iris_predictions_total', 'Total predictions made')

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)
    start_http_server(8001)  # Prometheus metrics on :8001

@app.post("/predict")
def predict(input: IrisInput):
    PREDICTION_COUNTER.inc()
    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}