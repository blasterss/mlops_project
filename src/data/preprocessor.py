from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import List

class DataPreprocessor:
    @staticmethod
    def _basic_clean(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Базовая очистка данных"""
        df = df.copy()
        
        df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], 
                axis=1, 
                inplace=True)
        
        mean_age = df.groupby('Sex')['Age'].mean().round()
        df['Age'] = df['Age'].fillna(df['Sex'].map(mean_age))
        df['Embarked'] = df['Embarked'].fillna('S')
        
        return df

    @staticmethod
    def _encode_categorical(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.get_dummies(df, drop_first=True, dtype=int)

    @staticmethod
    def build_preprocessor(
        numerical_cols: List[str]
    ) -> ColumnTransformer:
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        return ColumnTransformer([
            ('num', num_pipeline, numerical_cols)
        ])

    @staticmethod
    def full_preprocess(
        df: pd.DataFrame,
        # numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Полный пайплайн подготовки данных"""

        df_clean = DataPreprocessor._basic_clean(df)
        
        df_encoded = DataPreprocessor._encode_categorical(df_clean)
        
        # preprocessor = DataPreprocessor.build_preprocessor(numerical_cols)
        
        return df_encoded #, preprocessor