from .model import AbstractModel, RFCModel, GBCModel
from typing import Type, Dict
from sklearn.base import BaseEstimator

class ModelFactory:

    _model_registry: Dict[str, Type[AbstractModel]] = {
        'random_forest': RFCModel,
        'gradient_boosting': GBCModel
    }
        
    @staticmethod
    def create_model(model_name: str) -> AbstractModel:
        """Создание экземпляра модели"""
        try:
            return ModelFactory._model_registry[model_name]()
        except KeyError:
            available = list(ModelFactory._model_registry.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    @staticmethod
    def load_model_from_mlflow() -> BaseEstimator:
        """Загрузка обученной модели из MLflow"""
        import mlflow
        return mlflow.sklearn.load_model()