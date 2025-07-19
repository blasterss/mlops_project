from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from abc import ABC, abstractmethod
from optuna.trial import Trial
from typing import Dict, Any

class AbstractModel(ABC):
    """
    Abstract base class for machine learning models with hyperparameter tuning support.
    
    Provides common interface for:
    - Hyperparameter optimization with Optuna
    - Model training and prediction
    - Parameter logging to MLflow
    """
    
    def __init__(self):
        super().__init__()
        self.model = None  # Will be initialized in concrete subclasses

    @staticmethod
    @abstractmethod
    def get_parameters(trial: Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Optuna optimization.
        
        Args:
            trial: Optuna trial object for parameter suggestions
            
        Returns:
            Dictionary of parameter names and their suggested values
        """
        pass

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Apply hyperparameters to the model.
        
        Args:
            params: Dictionary of parameter names and values
        """
        self.model.set_params(**params)

    def fit(self, X, y) -> None:
        """
        Train the model on given data.
        
        Args:
            X: Feature matrix
            y: Target values
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Generate predictions for new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of predictions
        """
        return self.model.predict(X)

    def log_parameters(self, run) -> None:
        """
        Log all model parameters to MLflow.
        
        Args:
            run: MLflow active run
        """
        params = self.model.get_params()
        for param, value in params.items():
            run.log_param(param, value)


class RFCModel(AbstractModel):
    """
    Random Forest Classifier with Optuna hyperparameter optimization.
    
    Hyperparameter search space includes:
    - Tree structure parameters (max_depth, min_samples_split, etc.)
    - Ensemble parameters (n_estimators)
    - Feature selection parameters (max_features)
    """
    
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(random_state=42)

    @staticmethod
    def get_parameters(trial: Trial
                       ) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Random Forest.
        
        Returns:
            Dictionary of parameter suggestions including:
            - n_estimators: Number of trees (50-500)
            - max_depth: Maximum tree depth (3-15)
            - Various split/leaf constraints
        """
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
    """
    Gradient Boosting Classifier with Optuna hyperparameter optimization.
    
    Hyperparameter search space includes:
    - Boosting parameters (learning_rate, n_estimators)
    - Tree structure parameters
    - Regularization parameters
    - Early stopping configuration
    """
    
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier(random_state=42)

    @staticmethod
    def get_parameters(trial: Trial
                       ) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Gradient Boosting.
        
        Returns:
            Dictionary of parameter suggestions including:
            - learning_rate: Shrinkage factor (0.01-0.2)
            - n_estimators: Number of boosting stages (50-500)
            - Various regularization parameters
        """
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