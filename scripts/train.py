from src.utils.config import Config
from src.data.loader import DataLoader
from src.models.model_factory import ModelFactory
import mlflow

def train():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
if __name__ == "__main__":
    train()