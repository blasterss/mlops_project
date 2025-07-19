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
    """Класс для оптимизации гиперпараметров с помощью Optuna"""
    
    def __init__(self,
                 direction: str = "maximize",
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None,
                 load_if_exists: bool = True):
        """
        Args:
            direction: Направление оптимизации ("maximize" или "minimize")
            study_name: Имя исследования (для сохранения в storage)
            storage: Путь для сохранения исследований (например, "sqlite:///optuna.db")
            load_if_exists: Загружать существующее исследование если существует
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
        Запуск оптимизации
        
        Args:
            objective_func: Функция для оптимизации
            n_trials: Максимальное количество trials
            timeout: Максимальное время оптимизации в секундах
            callbacks: Список callback функций
        """
        callbacks = callbacks or []
        callbacks.append(self._log_progress_callback())
        
        try:
            self.study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=-1,
                callbacks=callbacks,
                gc_after_trial=True
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
        Создает функцию для оптимизации
        
        Args:
            model: Модель с методами get_parameters/set_parameters или sklearn-модель
            X: Признаки
            y: Целевая переменная
            scoring: Метрика для оптимизации
            cv: Количество фолдов для кросс-валидации
            
        Returns:
            Функция objective для optuna
        """
        def objective(trial: Trial):
            try:
                # Для совместимости с AbstractModel и sklearn
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
                
                scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
                return scores.mean()
                
            except Exception as e:
                logging.error(f"Trial failed: {str(e)}")
                raise optuna.TrialPruned()
                
        return objective

    def _log_progress_callback(self) -> Callable:
        """Callback для логирования прогресса"""
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
        """Возвращает модель с лучшими параметрами"""
        if hasattr(model, 'set_parameters'):
            model.set_parameters(self.study.best_params)
            return model
        return model.set_params(**self.study.best_params)