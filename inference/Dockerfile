FROM python:3.11-slim

WORKDIR /app

COPY inference_api.py .
COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8000

# Prometheus metrics
EXPOSE 8001  

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]