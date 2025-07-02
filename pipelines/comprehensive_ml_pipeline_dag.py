from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.sensors.filesystem import FileSensor
import sys
import os

# Add the scripts directory to the Python path
sys.path.append('/opt/airflow/scripts')

# Import our custom functions
from train import train_model
from compare_ab import compare_models
from rollback import rollback_model

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

# Define tasks
start_task = DummyOperator(
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

success_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_success_notification,
    dag=dag,
)

end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Define task dependencies
start_task >> train_task >> ab_test_task >> decision_task >> rollback_task >> success_task >> end_task
