# Titanic Survival Prediction ML Pipeline

An end-to-end machine learning pipeline for predicting Titanic passenger survival, featuring hyperparameter optimization with Optuna and experiment tracking with MLflow.

## 📌 Features

- **Data Processing**: Automated cleaning and feature engineering
- **Model Training**: 
  - Random Forest Classifier
  - Gradient Boosting Classifier
- **Hyperparameter Optimization**: Automated with Optuna
- **Experiment Tracking**: Full MLflow integration
- **Reproducible Pipeline**: Airflow DAG for workflow orchestration
- **Modular Design**: Clean separation of components

## 🛠️ Project Structure
```
mlops_project/
├── src/                        # Main source code
│ ├── __init__.py
│ ├── data/                     # Data block
│ │ ├── __init__.py  
│ │ ├── loader.py               # Data loading utilities
│ │ └── preprocessor.py         # Feature engineering
│ ├── models/                   # ML components
│ │ ├── __init__.py  
│ │ ├── model_factory.py        # Model creation
│ │ ├── model.py                # Model implementations
│ │ └── optuna_optimizer.py     # Hyperparameter tuning
│ └── utils/
│ │ ├── __init__.py  
│ │ └── config.py               # Path configuration
│ └── visualization/            # Visualization block (in development)
│ │ ├── __init__.py  
│ │ └── plotter.py              # For plotting (example barblot of feature's importance)
├── dags/
│ └── titanic_pipeline.py       # Airflow DAG definition
├── data/                       # Input data
│ │── gender_submission.csv     # Sample submission
│ │── test.csv                  # Test data
│ └── train.csv                 # Train data
├── notebooks/
│ └── exploring_data.ipynb      
├── scripts/                    # Utility scripts (in development)
│ │── test.py                   
│ └── train.py                 
├── submissions/                # Prediction outputs
├── mlruns/                     # MLflow tracking data (hidden)
├── airflow/                    # Airflow home directory (hidden)
└── README.md
```

## ⚙️ Configuration

Edit `src/utils/config.py` to configure:

```python
# Data paths
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA = DATA_DIR / "train.csv" 

# MLflow tracking
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 
MLFLOW_EXPERIMENT_NAME = "titanic_survival"
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-ml-pipeline.git
cd titanic-ml-pipeline
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export PYTHONPATH="${PYTHONPATH}:{path_to_project}"    
export AIRFLOW_HOME=$(pwd)/airflow
```

## Running pipeline

1. Initialize Airflow
```bash
airflow standalone
```
2. Start MLflow tracking server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

3. Access the Airflow web interface (typically at http://localhost:8080) and trigger the titanic_pipeline DAG.
