import pandas as pd
import joblib
from sklearn.metrics import classification_report
import os

def evaluate_model():
    data_path = os.environ.get('DATA_PATH', '/opt/airflow/dags/repo/data/iris.csv')  # <-- UPDATED
    model_path = os.environ.get('MODEL_PATH', '/opt/airflow/models/latest_model.pkl')
    df = pd.read_csv(data_path)
    X = df.drop('species', axis=1)
    y = df['species']
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
