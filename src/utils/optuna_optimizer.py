import optuna
from sklearn.model_selection import cross_val_score
import pandas as pd

class OptunaOptimizer:
    def __init__(self,
                 direction: str = "maximize",):
        self.study = optuna.create_study(direction=direction,
                                         sampler=optuna.samplers.TPESampler(seed=42),)
    
    def optimize(self, 
                 objective_func,
                 n_trials: int = 30,
                 timeout: int = 3600):
        self.study.optimize(objective_func, 
                            n_trials=n_trials, 
                            timeout=timeout, 
                            n_jobs=-1)
    
    @staticmethod
    def create_objective(model, 
                         X: pd.DataFrame, 
                         y: pd.Series):
        def objective(trial):
            params = model.get_parameters(trial)
            model.set_parameters(params)
            scores = cross_val_score(model.model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        return objective