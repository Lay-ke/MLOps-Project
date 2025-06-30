import pandas as pd
import joblib
from sklearn.metrics import classification_report

def evaluate_model():
    df = pd.read_csv('data/iris.csv')
    X = df.drop('species', axis=1)
    y = df['species']
    model = joblib.load('models/latest_model.pkl')
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_model()