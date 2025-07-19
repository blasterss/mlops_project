import optuna
from sklearn.model_selection import cross_val_score
from optuna.study import Study
from optuna.trial import Trial
from typing import Callable, Optional, Union
import pandas as pd
import logging
from .model import AbstractModel
from sklearn.base import BaseEstimator

class OptunaOptimizer:
    """
    A class for hyperparameter optimization using Optuna.
    
    Provides functionality to:
    - Perform hyperparameter optimization studies
    - Create objective functions for model tuning
    - Track optimization progress
    - Retrieve best performing models
    """
    
    def __init__(self,
                 direction: str = "maximize",
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None,
                 load_if_exists: bool = True):
        """
        Initialize the Optuna optimizer.
        
        Args:
            direction: Optimization direction ("maximize" or "minimize")
            study_name: Name of the study (for storage and tracking)
            storage: Database URL for storing studies (e.g., "sqlite:///optuna.db")
            load_if_exists: Whether to continue existing study if found
            
        Example:
            >>> optimizer = OptunaOptimizer(direction="maximize", study_name="rf_optimization")
        """
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=load_if_exists
        )
        self.logger = logging.getLogger(__name__)
        
    def optimize(self,
                 objective_func: Callable,
                 n_trials: int = 30,
                 timeout: Optional[int] = 3600,
                 callbacks: Optional[list] = None) -> None:
        """
        Run the hyperparameter optimization study.
        
        Args:
            objective_func: Objective function to optimize
            n_trials: Maximum number of trials to run
            timeout: Maximum time in seconds for optimization
            callbacks: List of callback functions to execute after each trial
            
        Raises:
            optuna.exceptions.StorageInternalError: If database connection fails
            RuntimeError: If optimization encounters critical errors
            
        Example:
            >>> optimizer.optimize(objective_func, n_trials=50, timeout=1800)
        """
        callbacks = callbacks or []
        callbacks.append(self._log_progress_callback())
        
        try:
            self.study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks
            )
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise

    @staticmethod
    def create_objective(model: Union[AbstractModel, BaseEstimator],
                        X: pd.DataFrame,
                        y: pd.Series,
                        scoring: str = 'accuracy',
                        cv: int = 3) -> Callable:
        """
        Create an objective function for Optuna optimization.
        
        Args:
            model: Model instance to optimize (AbstractModel or scikit-learn estimator)
            X: Feature matrix for training
            y: Target values
            scoring: Evaluation metric to optimize
            cv: Number of cross-validation folds
            
        Returns:
            A callable objective function for Optuna
            
        Example:
            >>> objective = optimizer.create_objective(model, X_train, y_train)
        """
        def objective(trial: Trial):
            try:
                # Handle both AbstractModel and sklearn estimators
                if hasattr(model, 'get_parameters'):
                    params = model.get_parameters(trial)
                    model.set_parameters(params)
                    estimator = model.model if hasattr(model, 'model') else model
                else:
                    params = {**model.get_params()}
                    for name, param in params.items():
                        if hasattr(param, '__optuna__'):
                            params[name] = param.suggest(trial)
                    estimator = model.set_params(**params)
                
                # Perform cross-validation
                scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
                return scores.mean()
                
            except Exception as e:
                logging.error(f"Trial failed: {str(e)}")
                raise optuna.TrialPruned()
                
        return objective

    def _log_progress_callback(self) -> Callable:
        """Create a callback function for logging optimization progress."""
        def callback(study: Study, 
                     trial: Trial):
            if study.best_trial.number == trial.number:
                self.logger.info(
                    f"New best trial {trial.number}: "
                    f"Value={trial.value:.4f}, "
                    f"Params={trial.params}"
                )
        return callback

    def get_best_model(self,
                       model: Union[AbstractModel, BaseEstimator]
                       ) -> Union[AbstractModel, BaseEstimator]:
        """
        Return a model instance configured with the best parameters found.
        
        Args:
            model: Model instance to configure with best parameters
            
        Returns:
            Model instance with optimized parameters
            
        Example:
            >>> best_model = optimizer.get_best_model(model)
        """
        if hasattr(model, 'set_parameters'):
            model.set_parameters(self.study.best_params)
            return model
        return model.set_params(**self.study.best_params)