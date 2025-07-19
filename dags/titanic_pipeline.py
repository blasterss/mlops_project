from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_factory import ModelFactory
from src.utils.config import Config
from src.models.optuna_optimizer import OptunaOptimizer

import pandas as pd

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1
}

def load_and_preprocess_data(is_train: bool, **context):
    """Обработка с проверкой данных"""
    ti = context['ti']

    try:
        df = DataLoader.load_data(is_train=is_train)
        df_preprocessed = DataPreprocessor.full_preprocess(df)
        ti.xcom_push(key='columns', value=list(df_preprocessed.columns))
        return df_preprocessed.to_dict('list')
    except Exception as e:
        ti.xcom_push(key='error', value=str(e))
        raise

def train_model_func(model_name: str, **context):
    """Обучение с обработкой ошибок"""
    ti = context['ti']
    
    import mlflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    train_data = ti.xcom_pull(task_ids='data_preparation.prep_train_data', key='return_value')
    columns = ti.xcom_pull(task_ids='data_preparation.prep_train_data', key='columns')
    df = pd.DataFrame(train_data, columns=columns)
    X, y = df.drop('Survived', axis=1), df['Survived']

    optimizer = OptunaOptimizer()
    
    try:
        model = ModelFactory.create_model(model_name)
        objective = optimizer.create_objective(model, X, y)
        optimizer.optimize(objective, n_trials=30)
        
        # Получение лучшей метрики
        best_accuracy = optimizer.study.best_value
        
        # Логирование
        with mlflow.start_run():
            mlflow.log_metric("val_accuracy", best_accuracy)
            mlflow.log_params(optimizer.study.best_params)
            mlflow.sklearn.log_model(optimizer.get_best_model(model), model_name)
            
        ti.xcom_push(key=f'{model_name}_accuracy', value=best_accuracy)
            
    except Exception as e:
        ti.xcom_push(key=f'{model_name}_error', value=str(e))
        raise
        
def get_submission_file(**context):
    """Генерация submission файла"""
    ti = context['ti']
    test_data = ti.xcom_pull(task_ids='data_preparation.prep_test_data', key='data')
    columns = ti.xcom_pull(task_ids='data_preparation.prep_test_data', key='columns')
    
    df_test = pd.DataFrame(test_data, columns=columns)
    
    # Получаем лучшую модель по accuracy
    rf_accuracy = ti.xcom_pull(task_ids='model_training.train_model_1', key='random_forest_accuracy')
    gb_accuracy = ti.xcom_pull(task_ids='model_training.train_model_2', key='gradient_boosting_accuracy')
    
    best_model_type = "random_forest" if rf_accuracy > gb_accuracy else "gradient_boosting"
    
    # Загружаем модель из MLflow
    import mlflow
    model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/{best_model_type}")
    
    predictions = model.predict(df_test)
    
    # Сохраняем submission.csv
    submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    
with DAG(
    "titanic_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=["mlops"]
) as dag:
    
    with TaskGroup("data_preparation") as data_prep_group:
        prep_train_data = PythonOperator(
            task_id="prep_train_data",
            python_callable=load_and_preprocess_data,
            op_args=[True]
        )
        
        prep_test_data = PythonOperator(
            task_id="prep_test_data",
            python_callable=load_and_preprocess_data,
            op_args=[False]
        )
    
    with TaskGroup("model_training") as train_group:
        train_model_1_task = PythonOperator(
            task_id="train_model_1",
            python_callable=train_model_func,
            op_args=["random_forest"],
        )
        
        train_model_2_task = PythonOperator(
            task_id="train_model_2",
            python_callable=train_model_func,
            op_args=["gradient_boosting"],
        )
    
    get_submission = PythonOperator(
        task_id="get_submission",
        python_callable=get_submission_file,
    )
    
    # Порядок выполнения задач
    data_prep_group >> train_group >> get_submission