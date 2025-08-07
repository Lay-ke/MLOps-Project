from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import os
from prometheus_client import start_http_server, Counter

app = FastAPI(title="Iris Classification API", description="ML model inference API for Iris species classification")
MODEL_NAME = "iris_classifier"
MODEL_ALIAS = "prod"  # Changed from MODEL_STAGE to MODEL_ALIAS

# Prometheus metrics
PREDICTION_COUNTER = Counter('iris_predictions_total', 'Total predictions made')
model = None  # Start with no model

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = 0.95  # Default confidence score

def get_tracking_uri():
    return os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')

def load_model():
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"  # Changed to use alias
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print("Model not available yet:", str(e))
        return None

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the web interface for iris classification"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Classification Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px;
            max-width: 500px;
            width: 100%;
            transition: transform 0.3s ease;
        }

        .container:hover {
            transform: translateY(-5px);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #4a5568;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: #718096;
            font-size: 1.1rem;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-group label {
            display: block;
            color: #4a5568;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 1rem;
        }

        .input-container {
            position: relative;
        }

        .form-group input {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: #f8fafc;
        }

        .form-group input:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            transform: translateY(-2px);
        }

        .unit {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #a0aec0;
            font-size: 0.9rem;
            pointer-events: none;
        }

        .predict-btn {
            width: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 18px 30px;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .predict-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }

        .predict-btn:active {
            transform: translateY(-1px);
        }

        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.5s ease;
        }

        .result.show {
            opacity: 1;
            transform: translateY(0);
        }

        .result.setosa {
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
        }

        .result.versicolor {
            background: linear-gradient(135deg, #ed8936, #dd6b20);
            color: white;
        }

        .result.virginica {
            background: linear-gradient(135deg, #9f7aea, #805ad5);
            color: white;
        }

        .result h3 {
            font-size: 1.5rem;
            margin-bottom: 10px;
        }

        .result p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .confidence {
            margin-top: 15px;
            font-size: 0.95rem;
            opacity: 0.8;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .iris-info {
            margin-top: 20px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }

        .iris-info h4 {
            color: #4a5568;
            margin-bottom: 10px;
        }

        .iris-info ul {
            color: #718096;
            padding-left: 20px;
        }

        .iris-info li {
            margin-bottom: 5px;
        }

        .status-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 600;
            z-index: 1000;
            transition: all 0.3s ease;
        }

        .status-indicator.online {
            background: #48bb78;
            color: white;
        }

        .status-indicator.offline {
            background: #f56565;
            color: white;
        }

        @media (max-width: 600px) {
            .container {
                padding: 30px 20px;
                margin: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="status-indicator" id="statusIndicator">🔄 Checking...</div>
    
    <div class="container">
        <div class="header">
            <h1>🌸 Iris Classifier</h1>
            <p>ML-powered iris species prediction</p>
        </div>

        <form id="predictionForm">
            <div class="form-group">
                <label for="sepalLength">Sepal Length</label>
                <div class="input-container">
                    <input type="number" id="sepalLength" step="0.1" min="0" max="10" required placeholder="e.g., 5.1">
                    <span class="unit">cm</span>
                </div>
            </div>

            <div class="form-group">
                <label for="sepalWidth">Sepal Width</label>
                <div class="input-container">
                    <input type="number" id="sepalWidth" step="0.1" min="0" max="10" required placeholder="e.g., 3.5">
                    <span class="unit">cm</span>
                </div>
            </div>

            <div class="form-group">
                <label for="petalLength">Petal Length</label>
                <div class="input-container">
                    <input type="number" id="petalLength" step="0.1" min="0" max="10" required placeholder="e.g., 1.4">
                    <span class="unit">cm</span>
                </div>
            </div>

            <div class="form-group">
                <label for="petalWidth">Petal Width</label>
                <div class="input-container">
                    <input type="number" id="petalWidth" step="0.1" min="0" max="10" required placeholder="e.g., 0.2">
                    <span class="unit">cm</span>
                </div>
            </div>

            <button type="submit" class="predict-btn" id="predictBtn">
                Predict Species
            </button>
        </form>

        <div id="result" class="result">
            <h3 id="species"></h3>
            <p id="description"></p>
            <div id="confidence" class="confidence"></div>
        </div>

        <div class="iris-info">
            <h4>Iris Species Information:</h4>
            <ul>
                <li><strong>Setosa:</strong> Small flowers, typically found in Alaska and Maine</li>
                <li><strong>Versicolor:</strong> Medium-sized flowers, common in eastern North America</li>
                <li><strong>Virginica:</strong> Large flowers, native to eastern United States</li>
            </ul>
        </div>
    </div>

    <script>
        const form = document.getElementById('predictionForm');
        const result = document.getElementById('result');
        const predictBtn = document.getElementById('predictBtn');
        const species = document.getElementById('species');
        const description = document.getElementById('description');
        const confidence = document.getElementById('confidence');
        const statusIndicator = document.getElementById('statusIndicator');

        const speciesInfo = {
            'setosa': {
                name: 'Iris Setosa',
                description: 'A beautiful iris with distinctive small petals and sepals',
                color: 'setosa'
            },
            'versicolor': {
                name: 'Iris Versicolor',
                description: 'A medium-sized iris commonly found in wetlands',
                color: 'versicolor'
            },
            'virginica': {
                name: 'Iris Virginica',
                description: 'A large, elegant iris with broad petals',
                color: 'virginica'
            }
        };

        // Check API health status
        async function checkHealth() {
            try {
                const response = await fetch('/health');
                const health = await response.json();
                
                if (health.status === 'ok' && health.model_loaded) {
                    statusIndicator.textContent = '✅ Model Ready';
                    statusIndicator.className = 'status-indicator online';
                } else {
                    statusIndicator.textContent = '⚠️ No Model';
                    statusIndicator.className = 'status-indicator offline';
                }
            } catch (error) {
                statusIndicator.textContent = '❌ Offline';
                statusIndicator.className = 'status-indicator offline';
            }
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const sepalLength = parseFloat(document.getElementById('sepalLength').value);
            const sepalWidth = parseFloat(document.getElementById('sepalWidth').value);
            const petalLength = parseFloat(document.getElementById('petalLength').value);
            const petalWidth = parseFloat(document.getElementById('petalWidth').value);

            // Show loading state
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<span class="loading"></span>Predicting...';
            result.classList.remove('show');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sepal_length: sepalLength,
                        sepal_width: sepalWidth,
                        petal_length: petalLength,
                        petal_width: petalWidth
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const prediction = await response.json();
                displayResult(prediction);
            } catch (error) {
                console.error('Prediction error:', error);
                displayError(error.message);
            } finally {
                predictBtn.disabled = false;
                predictBtn.innerHTML = 'Predict Species';
            }
        });

        function displayResult(prediction) {
            const predictedSpecies = prediction.prediction.toLowerCase();
            const info = speciesInfo[predictedSpecies];
            
            if (info) {
                species.textContent = info.name;
                description.textContent = info.description;
                confidence.textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
                
                result.className = `result ${info.color}`;
                result.classList.add('show');
            } else {
                displayError('Unknown species predicted');
            }
        }

        function displayError(errorMessage = 'Unable to make prediction') {
            species.textContent = 'Prediction Error';
            description.textContent = errorMessage;
            confidence.textContent = '';
            
            result.className = 'result';
            result.style.background = 'linear-gradient(135deg, #f56565, #e53e3e)';
            result.style.color = 'white';
            result.classList.add('show');
        }

        // Check health on load and set sample data
        window.addEventListener('load', () => {
            checkHealth();
            
            // Set sample data
            document.getElementById('sepalLength').value = '5.1';
            document.getElementById('sepalWidth').value = '3.5';
            document.getElementById('petalLength').value = '1.4';
            document.getElementById('petalWidth').value = '0.2';
            
            // Check health every 30 seconds
            setInterval(checkHealth, 30000);
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    start_http_server(8001)  # Prometheus metrics on :8001

@app.post("/predict", response_model=PredictionResponse)
def predict(input: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="No model available. Please train and promote a model.")
    
    PREDICTION_COUNTER.inc()
    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    prediction = model.predict(data)
    
    # Calculate a simple confidence score based on prediction probabilities if available
    confidence = 0.95  # Default confidence
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)[0]
            confidence = float(max(probabilities))
    except:
        pass
    
    return PredictionResponse(prediction=str(prediction[0]), confidence=confidence)

@app.post("/reload")
def reload_model():
    global model
    model = load_model()
    if model is None:
        return {"status": "no model available"}
    return {"status": "reloaded"}

@app.get("/health")
def health():
    """Enhanced health check endpoint with detailed status"""
    model_info = {}
    if model is not None:
        try:
            # Get model info if available
            model_info = {
                "model_type": type(model).__name__,
                "sklearn_version": hasattr(model, '__sklearn_version__'),
                "features": model.n_features_in_ if hasattr(model, 'n_features_in_') else 4
            }
        except:
            model_info = {"model_type": "unknown"}
    
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "model_info": model_info,
        "api_version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "reload": "/reload",
            "health": "/health",
            "interface": "/"
        }
    }

@app.get("/docs-simple")
async def get_simple_docs():
    """Simple API documentation"""
    return {
        "title": "Iris Classification API",
        "description": "ML model inference API for Iris species classification",
        "version": "1.0.0",
        "endpoints": {
            "/": "Web interface for interactive predictions",
            "/predict": "POST - Make predictions (requires JSON body with sepal_length, sepal_width, petal_length, petal_width)",
            "/health": "GET - Check API and model status",
            "/reload": "POST - Reload the ML model",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation"
        },
        "example_request": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "example_response": {
            "prediction": "setosa",
            "confidence": 0.95
        }
    }