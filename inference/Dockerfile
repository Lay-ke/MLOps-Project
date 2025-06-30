FROM python:3.10-slim

WORKDIR /app

COPY ../models ./models
COPY inference_api.py .
COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 8001  # Prometheus metrics

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]