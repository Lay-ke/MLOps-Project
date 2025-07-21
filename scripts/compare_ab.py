import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import os
import yaml
warnings.filterwarnings('ignore')

def get_tracking_uri():
    config_path = os.environ.get('CONFIG_PATH', '/opt/airflow/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('mlflow', {}).get('tracking_uri', 'http://mlflow-server:5000')



def get_latest_model_versions(model_name, num_versions=2):
    """
    Get the latest model versions from MLflow Model Registry
    
    Args:
        model_name (str): Name of the registered model
        num_versions (int): Number of latest versions to retrieve
    
    Returns:
        list: List of model versions sorted by version number (descending)
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get registered model details first
        registered_model = client.get_registered_model(model_name)
        
        # Get all versions using search_model_versions (newer API)
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        print(f"Found {len(model_versions)} total model versions")
        
        # Sort by version number in descending order
        model_versions = sorted(model_versions, key=lambda x: int(x.version), reverse=True)
        
        return model_versions[:num_versions]
    
    except Exception as e:
        print(f"Error retrieving model versions: {e}")
        return []

def load_model_from_mlflow(model_name, version):
    """
    Load a specific model version from MLflow
    
    Args:
        model_name (str): Name of the registered model
        version (str): Version number
    
    Returns:
        sklearn model: Loaded model
    """
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)

def evaluate_model(model, X, y):
    """
    Evaluate model performance with multiple metrics
    
    Args:
        model: Trained sklearn model
        X: Features
        y: True labels
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    predictions = model.predict(X)
    
    return {
        'accuracy': accuracy_score(y, predictions),
        'precision': precision_score(y, predictions, average='weighted'),
        'recall': recall_score(y, predictions, average='weighted'),
        'f1_score': f1_score(y, predictions, average='weighted')
    }

def compare_models(model_name="iris_classifier", data_path=None, test_split=0.3):
    """
    Compare the latest two versions of a model from MLflow Model Registry
    
    Args:
        model_name (str): Name of the registered model in MLflow
        data_path (str): Path to the test data
        test_split (float): Fraction of data to use for testing
    """
    print(f"\n=== A/B Model Comparison for {model_name} ===")
    
    # Load test data from S3
    try:
        data_path = data_path or os.environ.get('DATA_PATH', '/opt/aiflow/data/iris.csv')  # <-- UPDATED
        df = pd.read_csv(data_path)
        X = df.drop('species', axis=1)
        y = df['species']
        print(f"Loaded test data with {len(df)} samples")
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return
    
    # Get latest model versions
    model_versions = get_latest_model_versions(model_name, num_versions=2)
    
    if len(model_versions) < 2:
        print(f"\nWarning: Only {len(model_versions)} model version(s) found.")
        if len(model_versions) == 1:
            print("Only one model version available. Cannot perform A/B comparison.")
            print("Training a new model first to enable comparison...")
            return
        else:
            print("No model versions found. Please train a model first.")
            return
    
    print(f"\nFound {len(model_versions)} model versions for comparison:")
    for i, version in enumerate(model_versions):
        print(f"  Version {version.version}: Created {version.creation_timestamp}")
    
    # Load and evaluate models
    results = {}
    models = {}
    
    for i, version in enumerate(model_versions[:2]):
        version_num = version.version
        print(f"\nLoading Model Version {version_num}...")
        
        try:
            model = load_model_from_mlflow(model_name, version_num)
            models[f"v{version_num}"] = model
            
            # Evaluate model
            metrics = evaluate_model(model, X, y)
            results[f"v{version_num}"] = metrics
            
            print(f"Model v{version_num} Evaluation:")
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
                
        except Exception as e:
            print(f"Error loading/evaluating model version {version_num}: {e}")
    
    # Compare results
    result = {
        'metrics': results,
        'recommendation': None
    }
    if len(results) == 2:
        version_keys = list(results.keys())
        v1, v2 = version_keys[0], version_keys[1]
        
        print(f"\n=== Comparison Summary ===")
        print(f"{'Metric':<12} {'V' + v1[1:]:<12} {'V' + v2[1:]:<12} {'Winner':<10}")
        print("-" * 50)
        
        winners = {v1: 0, v2: 0}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            val1 = results[v1][metric]
            val2 = results[v2][metric]
            winner = v1 if val1 > val2 else v2 if val2 > val1 else "Tie"
            
            if winner != "Tie":
                winners[winner] += 1
            
            print(f"{metric.capitalize():<12} {val1:<12.4f} {val2:<12.4f} {winner.replace('v', 'V'):<10}")
        
        print("-" * 50)
        
        # Determine overall winner
        if winners[v1] > winners[v2]:
            overall_winner = v1.replace('v', 'Version ')
            print(f"\n🏆 Overall Winner: {overall_winner}")
        elif winners[v2] > winners[v1]:
            overall_winner = v2.replace('v', 'Version ')
            print(f"\n🏆 Overall Winner: {overall_winner}")
        else:
            print(f"\n🤝 Result: Tie between both versions")
        
        # Recommendation
        latest_version = model_versions[0].version
        print(f"\n📊 Recommendation:")
        if v1 == f"v{latest_version}" and winners[v1] >= winners[v2]:
            print(f"✅ Keep the latest version ({latest_version}) - it performs as well or better")
            result['recommendation'] = f"Keep the latest version ({latest_version})"
        elif v2 == f"v{latest_version}" and winners[v2] >= winners[v1]:
            print(f"✅ Keep the latest version ({latest_version}) - it performs as well or better")
            result['recommendation'] = f"Keep the latest version ({latest_version})"
        else:
            older_version = model_versions[1].version
            print(f"⚠️  Consider rolling back to version {older_version} - it shows better performance")
            result['recommendation'] = f"Rollback to version {older_version}"
    return result

if __name__ == "__main__":
    # Run A/B comparison
    compare_models()
