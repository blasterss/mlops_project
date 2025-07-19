from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_factory import ModelFactory
from src.utils.config import Config
from src.utils.mlflow_logger import MLflowLogger
from src.utils.optuna_optimizer import OptunaOptimizer

from typing import Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime
import mlflow

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

def split_data(**context) -> Dict[str, Any]:
    """Разделение данных с валидацией"""
    ti = context['ti']
    train_data = ti.xcom_pull(task_ids='data_preparation.prep_train_data', key='return_value')
    columns = ti.xcom_pull(task_ids='data_preparation.prep_train_data', key='columns')
    
    df_train = pd.DataFrame(train_data, columns=columns)
    
    X_train, X_val, y_train, y_val = DataLoader.split_data(df_train, 'Survived')
    
    return {
        'X_train': X_train.to_dict('list'),
        'X_val': X_val.to_dict('list'),
        'y_train': y_train.tolist(),
        'y_val': y_val.tolist()
    }

def train_model_func(model_name: str, **context):
    """Обучение с обработкой ошибок"""
    ti = context['ti']
    data = ti.xcom_pull(task_ids='split_data')

    logger = MLflowLogger("titanic_survival")
    optimizer = OptunaOptimizer()
    
    try:
        X_train = pd.DataFrame(data['X_train'])
        X_val = pd.DataFrame(data['X_val'])
        y_train = pd.Series(data['y_train'])
        y_val = pd.Series(data['y_val'])
        
        model = ModelFactory.create_model(model_name)
        objective = optimizer.create_objective(
            model,
            X_train, 
            y_train
        )
        optimizer.optimize(objective, n_trials=30)

        with logger.start_run(run_name=f"{model_name}_best"):
            best_model = ModelFactory.create_model(model_name)
            best_model.set_parameters(optimizer.study.best_params)
            best_model.fit(X_val, y_val)

            y_pred = best_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_pred)
            
            # Логирование
            best_model.log_to_mlflow()
            logger.log_study(optimizer.study)
            mlflow.log_metric("val_accuracy", val_accuracy)
            
            ti.xcom_push(key=f'{model_name}_accuracy', value=val_accuracy)
            
    except Exception as e:
        ti.xcom_push(key=f'{model_name}_error', value=str(e))
        raise

def load_best_model_from_mlflow(best_model_type: str):
    """Загрузка лучшей модели из MLflow"""
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

    runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.runName LIKE '%{best_model_type}%'",
        order_by=["start_time DESC"],
        max_results=1
    )

    model = None
    if not runs.empty:
        run_id = runs.iloc[0]['run_id']
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

    return model
        
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
    model = load_best_model_from_mlflow(best_model_type)

    predictions = model.predict(df_test)
    
    # Сохраняем submission.csv
    submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    
    ti.xcom_push(key='best_model', value=best_model_type)

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
    
    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
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
    data_prep_group >> split_data_task >> train_group >> get_submission