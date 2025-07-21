import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
import yaml
import argparse

def load_config(config_path=None):
    if not config_path:
        # Try environment variable first
        config_path = os.environ.get(
            'CONFIG_PATH',
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'config.yaml')
        )
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_tracking_uri():
    config_path = os.environ.get('CONFIG_PATH', '/opt/airflow/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('mlflow', {}).get('tracking_uri', 'http://mlflow-server:5000')

def get_experiment_name():
    config_path = os.environ.get('CONFIG_PATH', '/opt/airflow/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('mlflow', {}).get('experiment_name', 'Default')



def train_model(variant_name=None):
    """Train model with specified variant hyperparameters"""
    # Load configuration
    config = load_config()
    
    mlflow.set_tracking_uri(get_tracking_uri())
    # Get hyperparameters
    if variant_name and variant_name in config['hyperparameters']['variants']:
        hyperparams = config['hyperparameters']['variants'][variant_name]
        print(f"Training with variant '{variant_name}': {hyperparams}")
    else:
        hyperparams = config['hyperparameters']['base']
        print(f"Training with base hyperparameters: {hyperparams}")
    
    # Load and prepare data from S3
    data_path = os.environ.get('DATA_PATH', '/opt/airflow/dags/repo/data/iris.csv')  # <-- UPDATED
    df = pd.read_csv(data_path)
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment(get_experiment_name())
    with mlflow.start_run():
        # Create RandomForestClassifier with dynamic hyperparameters
        clf = RandomForestClassifier(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            min_samples_split=hyperparams['min_samples_split'],
            min_samples_leaf=hyperparams['min_samples_leaf'],
            max_features=hyperparams['max_features'],
            random_state=42  # Keep random_state fixed for reproducibility
        )
        
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("variant", variant_name if variant_name else "base")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        
        # Create model signature
        signature = infer_signature(X_train, clf.predict(X_train))
        
        # Log model with signature and name
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:5],  # First 5 rows as example
            registered_model_name="iris_classifier"
        )
        
        # Save model locally
        os.makedirs('/opt/airflow/models', exist_ok=True)
        model_filename = f'latest_model_{variant_name}.pkl' if variant_name else 'latest_model.pkl'
        joblib.dump(clf, f'/opt/airflow/models/{model_filename}')
        
        variant_info = f" (variant: {variant_name})" if variant_name else ""
        print(f"Model trained with accuracy: {acc:.4f}{variant_info}")
        print(f"Hyperparameters used: {hyperparams}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Iris classifier with configurable hyperparameters')
    parser.add_argument('--variant', type=str, help='Hyperparameter variant to use (e.g., deep_forest, shallow_wide, conservative)')
    
    args = parser.parse_args()
    train_model(variant_name=args.variant)
