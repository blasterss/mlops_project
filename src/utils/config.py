import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    SUBMISSION_DIR = BASE_DIR / "submissions"

    TRAIN_DATA = DATA_DIR / "train.csv"
    TEST_DATA = DATA_DIR / "test.csv"
    SUBMISSION_SOURCE = DATA_DIR / "gender_submission.csv"

    SUBMISSION_DEST = SUBMISSION_DIR / "my_submission.csv"

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    MLFLOW_EXPERIMENT_NAME = "titanic_survival"