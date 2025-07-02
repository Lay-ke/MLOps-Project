import mlflow
import os
import yaml

def get_tracking_uri():
    config_path = os.environ.get('CONFIG_PATH', '/opt/airflow/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('mlflow', {}).get('tracking_uri', 'http://mlflow:5000')

mlflow.set_tracking_uri(get_tracking_uri())

def list_experiments():
    for exp in mlflow.search_experiments():
        print(f"Name: {exp.name}, ID: {exp.experiment_id}")

if __name__ == "__main__":
    list_experiments()