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