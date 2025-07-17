import pandas as pd
from ..utils.config import Config

class DataLoader:
    
    @staticmethod
    def load_train_data() -> pd.DataFrame:
        """Load train dataset"""
        return pd.read_csv(Config.TRAIN_DATA)
    
    @staticmethod
    def load_test_data() -> pd.DataFrame:
        """Load test dataset"""
        return pd.read_csv(Config.TEST_DATA)
    
    @staticmethod
    def split_data(
        df: pd.DataFrame, 
        target: str,
        test_size: float = 0.2
    ):
        """Split data into train/test"""
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=42)