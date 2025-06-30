import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime

# Set MLflow tracking URI to match other scripts
mlflow.set_tracking_uri("file:///home/yaw/Documents/LABS_HUB/MlOps-Project/mlruns")

def rollback_model(model_name="iris_classifier"):
    """
    Rollback the model by archiving the current production version
    and promoting the previous version to production.
    """
    client = MlflowClient()
    
    try:
        # Get all versions of the model sorted by version number (descending)
        model_versions = client.search_model_versions(
            filter_string=f"name='{model_name}'",
            order_by=["version_number DESC"]
        )
        
        if len(model_versions) < 2:
            print(f"Error: Need at least 2 model versions for rollback. Found {len(model_versions)} versions.")
            return False
        
        # Find current production version and previous version
        current_prod_version = None
        previous_version = None
        
        for version in model_versions:
            if version.current_stage == "Production":
                current_prod_version = version
            elif version.current_stage in ["Staging", "None"] and previous_version is None:
                # Get the highest version that's not in production
                if current_prod_version is None or int(version.version) < int(current_prod_version.version):
                    previous_version = version
        
        # If no production version exists, use the latest as current and second latest as previous
        if current_prod_version is None:
            current_prod_version = model_versions[0]  # Latest version
            previous_version = model_versions[1] if len(model_versions) > 1 else None
        
        if previous_version is None:
            print("Error: Could not find a previous version to rollback to.")
            return False
        
        print(f"Rolling back from version {current_prod_version.version} to version {previous_version.version}")
        
        # Archive the current production version
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod_version.version,
            stage="Archived",
            archive_existing_versions=False
        )
        
        # Promote the previous version to production
        client.transition_model_version_stage(
            name=model_name,
            version=previous_version.version,
            stage="Production",
            archive_existing_versions=False
        )
        
        # Log the rollback event
        with mlflow.start_run(run_name=f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("action", "rollback")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("rolled_back_from_version", current_prod_version.version)
            mlflow.log_param("rolled_back_to_version", previous_version.version)
            mlflow.log_param("rollback_timestamp", datetime.now().isoformat())
            
            print(f"Rollback complete: Model {model_name} rolled back from version {current_prod_version.version} to version {previous_version.version}")
            print(f"Version {current_prod_version.version} archived, version {previous_version.version} promoted to Production")
        
        return True
        
    except Exception as e:
        print(f"Error during rollback: {str(e)}")
        return False

def get_production_model(model_name="iris_classifier"):
    """
    Load the current production model from MLflow Model Registry.
    """
    client = MlflowClient()
    
    try:
        # Get the production version
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not prod_versions:
            print(f"No production version found for model {model_name}")
            return None
        
        prod_version = prod_versions[0]
        print(f"Loading production model: {model_name} version {prod_version.version}")
        
        # Load the model
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model, prod_version
        
    except Exception as e:
        print(f"Error loading production model: {str(e)}")
        return None

def test_production_model(model_name="iris_classifier"):
    """
    Test the current production model with sample data.
    """
    result = get_production_model(model_name)
    if result is None:
        return False
    
    model, version_info = result
    
    # Test with sample Iris data
    test_data = pd.DataFrame({
        'sepal_length': [5.1, 6.2, 7.3],
        'sepal_width': [3.5, 3.4, 2.9],
        'petal_length': [1.4, 4.5, 6.3],
        'petal_width': [0.2, 1.5, 1.8]
    })
    
    try:
        predictions = model.predict(test_data)
        print(f"Production model (version {version_info.version}) test predictions: {predictions}")
        return True
    except Exception as e:
        print(f"Error testing production model: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing current production model...")
        test_production_model()
    else:
        print("Performing model rollback...")
        success = rollback_model()
        if success:
            print("\nTesting rolled back model...")
            test_production_model()
