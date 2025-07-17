from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from abc import ABC, abstractmethod
from optuna.trial import Trial
from typing import Dict, Any

class AbstractModel(ABC):
    """Abstract base class for all ML models with hyperparameter tuning."""
    
    def __init__(self):
        super().__init__()
        self.model = None

    @staticmethod
    @abstractmethod
    def get_parameters(trial: Trial
                       ) -> Dict[str, Any]:
        """Get hyperparameters for the model from Optuna trial."""
        pass

    def set_parameters(self, params: Dict[str, Any]
                       ) -> None:
        """Set hyperparameters for the model."""
        self.model.set_params(**params)

    def fit(self, X, y
            ) -> None:
        """Train the model on given data."""
        self.model.fit(X, y)

    def predict(self, X
                ):
        """Make predictions on new data."""
        return self.model.predict(X)

    def log_parameters(self, run
                       ) -> None:
        """Log model parameters to MLflow."""
        params = self.model.get_params()
        for param, value in params.items():
            run.log_param(param, value)

class RFCModel(AbstractModel):
    """Random Forest Classifier with Optuna hyperparameter optimization."""
    
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()

    @staticmethod
    def get_parameters(trial: Trial
                       ) -> Dict[str, Any]:
        """Get hyperparameters for RandomForest from Optuna trial."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }

class GBCModel(AbstractModel):
    """Gradient Boosting Classifier with Optuna hyperparameter optimization."""
    
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier()

    @staticmethod
    def get_parameters(trial: Trial
                       ) -> Dict[str, Any]:
        """Get hyperparameters for GradientBoosting from Optuna trial."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'loss': trial.suggest_categorical('loss', ['log_loss', 'exponential']),
            'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.2),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
        }