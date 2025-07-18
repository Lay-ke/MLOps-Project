from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
import sys
import os
import requests

# Add the scripts directory to the Python path
sys.path.append('/opt/airflow/scripts')

# Import our custom functions
from train import train_model
from compare_ab import compare_models
from rollback import rollback_model, promote_model_to_production

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'comprehensive_ml_pipeline',
    default_args=default_args,
    description='Comprehensive ML pipeline with training, A/B testing, and automated rollback',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    max_active_runs=1,
)

def train_with_hyperparameter_variant(**context):
    """Train model with a specific hyperparameter variant"""
    # You can make this dynamic by passing variant name as parameter
    variant_name = context.get('dag_run').conf.get('variant', 'deep_forest')
    print(f"Training model with variant: {variant_name}")
    
    # Change to project directory
    os.chdir('/opt/airflow')
    
    # Train the model
    train_model(variant_name)
    return f"Model trained successfully with variant: {variant_name}"

def perform_ab_testing(**context):
    """Perform A/B testing between latest model versions"""
    print("Starting A/B testing...")
    
    # Change to project directory
    os.chdir('/opt/airflow')
    
    # Load and compare models
    result = compare_models()
    
    # Store result in XCom for downstream tasks
    context['task_instance'].xcom_push(key='ab_test_result', value=result)
    return result

def make_rollback_decision(**context):
    """Decide whether to rollback based on A/B test results"""
    # Get A/B test results from XCom
    ab_result = context['task_instance'].xcom_pull(key='ab_test_result', task_ids='ab_testing')
    
    print(f"A/B Test Result: {ab_result}")
    
    # Decision logic: rollback if latest model performs worse
    if ab_result and 'recommendation' in ab_result:
        recommendation = ab_result['recommendation']
        
        # If recommendation suggests previous model is better, trigger rollback
        if 'previous' in recommendation.lower() or 'rollback' in recommendation.lower():
            print("Decision: ROLLBACK - Previous model performs better")
            context['task_instance'].xcom_push(key='rollback_decision', value='rollback')
            return 'rollback'
        else:
            print("Decision: KEEP - Latest model performs well")
            context['task_instance'].xcom_push(key='rollback_decision', value='keep')
            return 'keep'
    else:
        print("Decision: KEEP - Unable to compare models or insufficient data")
        context['task_instance'].xcom_push(key='rollback_decision', value='keep')
        return 'keep'

def execute_rollback(**context):
    """Execute rollback if needed"""
    decision = context['task_instance'].xcom_pull(key='rollback_decision', task_ids='make_decision')
    
    if decision == 'rollback':
        print("Executing rollback...")
        os.chdir('/opt/airflow')
        rollback_model()
        return "Rollback executed successfully"
    else:
        print("No rollback needed")
        return "No rollback needed"

def send_success_notification(**context):
    """Send success notification"""
    decision = context['task_instance'].xcom_pull(key='rollback_decision', task_ids='make_decision')
    ab_result = context['task_instance'].xcom_pull(key='ab_test_result', task_ids='ab_testing')
    
    if decision == 'rollback':
        message = "ML Pipeline completed with ROLLBACK executed due to poor model performance"
    else:
        message = "ML Pipeline completed successfully - Latest model performing well"
    
    print(f"SUCCESS NOTIFICATION: {message}")
    print(f"A/B Test Details: {ab_result}")
    return message

def promote_latest_model_task(**context):
    print("Promoting latest model to Production...")
    promote_model_to_production()
    return "Latest model promoted to Production"

def trigger_inference_reload(**context):
    inference_url = os.environ.get("INFERENCE_API_URL", "http://inference-api:8000/reload")
    try:
        response = requests.post(inference_url)
        response.raise_for_status()
        print("Inference API reload triggered:", response.json())
        return response.json()
    except Exception as e:
        print("Failed to trigger inference reload:", str(e))
        raise

# Define tasks
start_task = EmptyOperator(
    task_id='start_pipeline',
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_with_hyperparameter_variant,
    dag=dag,
)

ab_test_task = PythonOperator(
    task_id='ab_testing',
    python_callable=perform_ab_testing,
    dag=dag,
)

decision_task = PythonOperator(
    task_id='make_decision',
    python_callable=make_rollback_decision,
    dag=dag,
)

rollback_task = PythonOperator(
    task_id='execute_rollback',
    python_callable=execute_rollback,
    dag=dag,
)

reload_inference_task = PythonOperator(
    task_id='reload_inference_api',
    python_callable=trigger_inference_reload,
    dag=dag,
)

success_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_success_notification,
    dag=dag,
)

end_task = EmptyOperator(
    task_id='end_pipeline',
    dag=dag,
)

promote_task = PythonOperator(
    task_id='promote_latest_model',
    python_callable=promote_latest_model_task,
    dag=dag,
)

# Define task dependencies
start_task >> train_task >> ab_test_task >> decision_task
# If rollback is needed, execute rollback, else promote latest model
rollback_task.set_upstream(decision_task)
promote_task.set_upstream(decision_task)
rollback_task >> reload_inference_task >> success_task >> end_task
promote_task >> reload_inference_task
