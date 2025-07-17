from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.config import Config
from src.models.model_factory import ModelFactory

from airflow.sdk import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import datetime


with DAG(
    "first_dag",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:
    EmptyOperator(task_id="task")