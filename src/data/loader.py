import pandas as pd
from ..utils.config import Config

class DataLoader:
    
    @staticmethod
    def load_data(train: bool = True) -> pd.DataFrame:
        """Load train or test dataset"""
        return pd.read_csv(Config.TRAIN_DATA) if train else pd.read_csv(Config.TEST_DATA)
    
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