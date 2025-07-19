from .model import AbstractModel, RFCModel, GBCModel
from typing import Type, Dict

class ModelFactory:
    """
    Factory class for creating and managing machine learning models.
    
    Provides functionality to:
    - Create model instances by name
    """
    
    # Registry of available models mapping names to model classes
    _model_registry: Dict[str, Type[AbstractModel]] = {
        'random_forest': RFCModel,
        'gradient_boosting': GBCModel
    }
        
    @staticmethod
    def create_model(model_name: str) -> AbstractModel:
        """
        Create a new instance of the specified model.
        
        Args:
            model_name: Name of the model to create (must be in registry)
            
        Returns:
            Instance of the requested model class
            
        Raises:
            ValueError: If the model name is not found in registry
            
        Example:
            >>> model = ModelFactory.create_model('random_forest')
        """
        try:
            return ModelFactory._model_registry[model_name]()
        except KeyError as e:
            available_models = list(ModelFactory._model_registry.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available_models}"
            ) from e
