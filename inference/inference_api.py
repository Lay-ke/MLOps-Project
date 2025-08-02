from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import os
from prometheus_client import start_http_server, Counter

app = FastAPI()
MODEL_NAME = os.environ.get("MODEL_NAME", "iris_classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "prod")  # NEW: use alias instead of stage

# Prometheus metrics
PREDICTION_COUNTER = Counter('iris_predictions_total', 'Total predictions made')
model = None  # Global model

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def get_tracking_uri():
    return os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')

def load_model():
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        print(f"Loading model from URI: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print("Model not available:", str(e))
        return None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    start_http_server(8001)  # Prometheus metrics

@app.post("/predict")
def predict(input: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="No model available. Please promote a model to alias.")
    PREDICTION_COUNTER.inc()
    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    prediction = model.predict(data)
    return {"prediction": str(prediction[0])}

@app.post("/reload")
def reload_model():
    global model
    model = load_model()
    if model is None:
        return {"status": "no model available"}
    return {"status": "reloaded"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
