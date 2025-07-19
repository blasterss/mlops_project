import mlflow
from optuna import Study
import optuna.visualization as vis
from typing import Dict, Any

class MLflowLogger:
    def __init__(self,
                 experiment_name: str = "default"):
        mlflow.set_experiment(experiment_name)
    
    def start_run(self,
                  run_name: str = None,
                  nested: bool = False):
        return mlflow.start_run(run_name=run_name, nested=nested)
    
    def log_study(self,
                  study: Study):
        """Логирование всего исследования Optuna"""
        fig = vis.plot_optimization_history(study)
        mlflow.log_figure(fig, "optimization_history.png")
        
        fig = vis.plot_param_importances(study)
        mlflow.log_figure(fig, "param_importances.png")
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_score", study.best_value)