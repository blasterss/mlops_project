import pandas as pd
from ..utils.config import Config

class DataLoader:
    
    @staticmethod
    def load_data(is_train: bool = True) -> pd.DataFrame:
        """Загрузка тренировочных и тестовых данных"""
        return pd.read_csv(Config.TRAIN_DATA) if is_train else pd.read_csv(Config.TEST_DATA)
    
    @staticmethod
    def load_sub_data() -> pd.DataFrame:
        """Загрузка примера submission файда"""
        return pd.read_csv(Config.SUBMISSION_SOURCE)
    
    @staticmethod
    def split_data(
        df: pd.DataFrame, 
        target: str,
        test_size: float = 0.2
    ):
        """Разделение данных на тренировочную/тестовую выборки"""
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=42)