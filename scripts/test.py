from src.utils.config import Config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
import mlflow

def test(model_name: str):
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    data = DataLoader.load_data(is_train=False)
    df = DataPreprocessor.full_preprocess(data)

    runs = mlflow.search_runs(
        experiment_names=[Config.MLFLOW_EXPERIMENT_NAME],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError("No runs found")
    
    last_run_id = runs.iloc[0].run_id
    model = mlflow.sklearn.load_model(f"runs:/{last_run_id}/{model_name}")
    
    predictions = model.predict(df)
    
    submission = DataLoader.load_sub_data()
    submission['Survived'] = predictions
    submission.to_csv(Config.SUBMISSION_DEST, index=False)

if __name__ == "__main__":
    test(model_name="random_forest")