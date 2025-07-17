from sklearn.ensemble import RandomForestClassifier

import mlflow

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("train_model")

model = RandomForestClassifier()

with mlflow.start_run():
    
    mlflow.sklearn.log_model(
        model,
        artifact_path="rfc",
        registered_model_name="rfc"
    )
    
    mlflow.log_artifact(local_path="",
                        artifact_path="train_model code")
    
    mlflow.end_run()

model.fit()

from src.utils.config import Config
from src.data.loader import DataLoader
from src.models.model_factory import ModelFactory
import mlflow

def main():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    # Весь pipeline из DAG, но в виде скрипта
    # ...

if __name__ == "__main__":
    main()