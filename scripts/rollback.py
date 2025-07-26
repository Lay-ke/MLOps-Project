import os
import yaml
from datetime import datetime

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def get_tracking_uri():
    config_path = os.environ.get('CONFIG_PATH', '/opt/airflow/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('mlflow', {}).get('tracking_uri', 'http://mlflow-server:5000')


mlflow.set_tracking_uri(get_tracking_uri())
client = MlflowClient()


def get_latest_model_version(model_name):
    versions = client.search_model_versions(f"name='{model_name}'", order_by=["version_number DESC"])
    return versions[0] if versions else None


def get_model_version_by_alias(model_name, alias_name="prod"):
    try:
        return client.get_model_version_by_alias(model_name, alias_name)
    except Exception:
        return None


def promote_model_to_alias(model_name="iris_classifier", alias_name="prod"):
    latest = get_latest_model_version(model_name)
    if not latest:
        print(f"No versions found for model '{model_name}'")
        return False

    try:
        current = get_model_version_by_alias(model_name, alias_name)
        if current and current.version != latest.version:
            print(f"Removing alias '{alias_name}' from version {current.version}")
            client.delete_registered_model_alias(name=model_name, alias=alias_name)

        client.set_registered_model_alias(name=model_name, alias=alias_name, version=latest.version)
        print(f"Model '{model_name}' version {latest.version} promoted to alias '{alias_name}'")
        return True
    except Exception as e:
        print("Promotion error:", str(e))
        return False


def rollback_model(model_name="iris_classifier", alias_name="prod"):
    versions = client.search_model_versions(f"name='{model_name}'", order_by=["version_number DESC"])
    if len(versions) < 2:
        print("Rollback failed: need at least 2 model versions.")
        return {'status': 'error', 'message': 'At least 2 versions required'}

    current = get_model_version_by_alias(model_name, alias_name)
    if not current:
        current = versions[0]
    prev = None

    for version in versions:
        if version.version != current.version:
            prev = version
            break

    if not prev:
        print("Rollback failed: no previous version found.")
        return {'status': 'error', 'message': 'No previous version found'}

    try:
        client.delete_registered_model_alias(name=model_name, alias=alias_name)
        client.set_registered_model_alias(name=model_name, alias=alias_name, version=prev.version)

        with mlflow.start_run(run_name=f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params({
                "action": "rollback",
                "model_name": model_name,
                "from_version": current.version,
                "to_version": prev.version,
                "timestamp": datetime.now().isoformat()
            })

        print(f"Rolled back: {model_name} from v{current.version} to v{prev.version}")
        return {'status': 'success', 'from_version': current.version, 'to_version': prev.version}

    except Exception as e:
        print("Rollback error:", str(e))
        return {'status': 'error', 'message': str(e)}


def get_model_by_alias(model_name="iris_classifier", alias_name="prod"):
    try:
        version_info = get_model_version_by_alias(model_name, alias_name)
        if not version_info:
            print(f"No version found for alias '{alias_name}'")
            return None

        model_uri = f"models:/{model_name}@{alias_name}"
        model = mlflow.sklearn.load_model(model_uri)
        return model, version_info

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def test_model_by_alias(model_name="iris_classifier", alias_name="prod"):
    result = get_model_by_alias(model_name, alias_name)
    if not result:
        return False

    model, version_info = result
    test_data = pd.DataFrame({
        'sepal_length': [5.1, 6.2, 7.3],
        'sepal_width': [3.5, 3.4, 2.9],
        'petal_length': [1.4, 4.5, 6.3],
        'petal_width': [0.2, 1.5, 1.8]
    })

    try:
        predictions = model.predict(test_data)
        print(f"Test passed. Model v{version_info.version} predictions: {predictions}")
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            print("Testing model by alias...")
            test_model_by_alias()
        elif sys.argv[1] == "promote":
            print("Promoting latest model...")
            promote_model_to_alias()
        else:
            print("Unknown command")
    else:
        print("Performing model rollback...")
        result = rollback_model()
        if result['status'] == 'success':
            print("Testing rolled back model...")
            test_model_by_alias()
