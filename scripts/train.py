from src.utils.config import Config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_factory import ModelFactory
from src.models.optuna_optimizer import OptunaOptimizer
import mlflow
from mlflow.models.signature import infer_signature

def train(model_name: str):
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    data = DataLoader.load_data(is_train=True)
    df = DataPreprocessor.full_preprocess(data)

    X, y = df.drop('Survived', axis=1), df['Survived']

    optimizer = OptunaOptimizer()

    model = ModelFactory.create_model(model_name)
    
    objective = optimizer.create_objective(model, X, y)
    optimizer.optimize(objective)

    with mlflow.start_run():
        best_model = optimizer.get_best_model(model)
        best_model.fit(X, y)

        best_accuracy = optimizer.study.best_value
        predictions = best_model.predict(X)

        mlflow.log_metric("val_accuracy", best_accuracy)
        mlflow.log_params(optimizer.study.best_params)

        signature = infer_signature(X, predictions)
        mlflow.sklearn.log_model(
            best_model, 
            name=model_name, 
            signature=signature,
            input_example=X.iloc[:1]
        )

if __name__ == "__main__":
    train(model_name="random_forest")