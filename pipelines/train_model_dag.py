from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.append('/opt/airflow/scripts')
from train import train_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'iris_train_model',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )