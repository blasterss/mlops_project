from .model import RFCModel, GBCModel

class ModelFactory:
    @staticmethod
    def create_model(model_type: str):
        """Factory method to create models"""
        models = {
            'random_forest': RFCModel,
            'gradient_boosting': GBCModel
        }
        return models[model_type]()