from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_factory import ModelFactory
from src.utils.config import Config
from src.models.optuna_optimizer import OptunaOptimizer

import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1
}

def load_and_preprocess_data(is_train: bool, **context):
    """
    Load and preprocess Titanic dataset
    
    Args:
        is_train (bool): Whether to load training data (True) or test data (False)
        context: Airflow context dictionary
        
    Returns:
        dict: Preprocessed data in dictionary format
    """
    ti = context['ti']

    try:
        # Load and preprocess data
        df = DataLoader.load_data(is_train=is_train)
        df_preprocessed = DataPreprocessor.full_preprocess(df)

        # Push column names and data to XCom
        ti.xcom_push(key='columns', value=list(df_preprocessed.columns))
        return df_preprocessed.to_dict('list')
    except Exception as e:
        ti.xcom_push(key='error', value=str(e))
        raise

def train_model_func(model_name: str, **context):
    """
    Train a model with Optuna optimization and log to MLflow
    
    Args:
        model_name (str): Name of model to train ('random_forest' or 'gradient_boosting')
        context: Airflow context dictionary
    """
    ti = context['ti']

    # Retrieve preprocessed training data
    train_data = ti.xcom_pull(task_ids='data_preparation.prep_train_data', key='return_value')
    columns = ti.xcom_pull(task_ids='data_preparation.prep_train_data', key='columns')

    # Prepare features and target
    df = pd.DataFrame(train_data, columns=columns)
    X, y = df.drop('Survived', axis=1), df['Survived']

    # Initialize Optuna optimizer
    optimizer = OptunaOptimizer()
    
    try:
        # Create and optimize model
        model = ModelFactory.create_model(model_name)
        objective = optimizer.create_objective(model, X, y)
        optimizer.optimize(objective, n_trials=30)
        
        # MLflow logging
        with mlflow.start_run() as run:
            # Get and fit best model
            best_model = optimizer.get_best_model(model)
            best_model.fit(X, y)

            # Calculate metrics
            best_accuracy = optimizer.study.best_value
            predictions = best_model.predict(X)

            # Log metrics and parameters
            mlflow.log_metric("val_accuracy", best_accuracy)
            mlflow.log_params(optimizer.study.best_params)

            # Create and log model signature
            signature = infer_signature(X, predictions)
            mlflow.sklearn.log_model(
                best_model, 
                name=model_name, 
                signature=signature,
                input_example=X.iloc[:1]
            )

            # Push run info to XCom
            ti.xcom_push(key=f"{model_name}_run_id", value=run.info.run_id)
            ti.xcom_push(key=f'{model_name}_accuracy', value=best_accuracy)
            
    except Exception as e:
        ti.xcom_push(key=f'{model_name}_error', value=str(e))
        raise
        
def get_submission_file(**context):
    """
    Generate submission file using the best model
    
    Args:
        context: Airflow context dictionary
    """
    ti = context['ti']

    # Retrieve test data
    test_data = ti.xcom_pull(task_ids='data_preparation.prep_test_data', key='return_value')
    columns = ti.xcom_pull(task_ids='data_preparation.prep_test_data', key='columns')
    df_test = pd.DataFrame(test_data, columns=columns)
    
    # Get model accuracies
    rf_accuracy = ti.xcom_pull(task_ids='model_training.train_model_1', key='random_forest_accuracy')
    gb_accuracy = ti.xcom_pull(task_ids='model_training.train_model_2', key='gradient_boosting_accuracy')
    
    # Get run IDs
    rf_run_id = ti.xcom_pull(task_ids='model_training.train_model_1', key='random_forest_run_id')
    gb_run_id = ti.xcom_pull(task_ids='model_training.train_model_2', key='gradient_boosting_run_id')
    
    # Load best model
    if rf_accuracy > gb_accuracy:
        model = mlflow.sklearn.load_model(f"runs:/{rf_run_id}/random_forest")
    else:
        model = mlflow.sklearn.load_model(f"runs:/{gb_run_id}/gradient_boosting")
    
    # Generate predictions
    predictions = model.predict(df_test)
    
    # Save submission file
    submission = DataLoader.load_sub_data()
    submission['Survived'] = predictions
    submission.to_csv(Config.SUBMISSION_DEST, index=False)

# Define the DAG
with DAG(
    "titanic_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=["mlops"]
) as dag:
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    # Data preparation task group
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
    
    # Model training task group
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
    
    # Submission generation task
    get_submission = PythonOperator(
        task_id="get_submission",
        python_callable=get_submission_file,
    )
    
    # Define task dependencies
    data_prep_group >> train_group >> get_submission