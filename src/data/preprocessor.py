from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import List

class DataPreprocessor:
    """
    A class for preprocessing Titanic dataset with methods for:
    - Basic data cleaning
    - Categorical encoding
    - Building sklearn preprocessing pipelines
    - Full preprocessing workflow
    """
    
    @staticmethod
    def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning operations:
        - Remove non-predictive columns
        - Handle missing values
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Cleaned DataFrame with:
            - Removed columns: PassengerId, Name, Cabin, Ticket
            - Age imputed by sex group mean
            - Embarked missing values filled with 'S'
        """
        df = df.copy()
        
        # Remove columns with low predictive value
        cols_to_drop = ['PassengerId', 'Name', 'Cabin', 'Ticket']
        df.drop(cols_to_drop, axis=1, inplace=True)
        
        # Impute missing age values with sex-grouped means
        mean_age = df.groupby('Sex')['Age'].mean().round()
        df['Age'] = df['Age'].fillna(df['Sex'].map(mean_age))
        
        # Fill missing Embarked values with most common value
        df['Embarked'] = df['Embarked'].fillna('S')
        
        return df

    @staticmethod
    def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with categorical variables encoded as dummy variables
            (first category dropped to avoid multicollinearity)
        """
        return pd.get_dummies(df, drop_first=True, dtype=int)

    @staticmethod
    def build_preprocessor(numerical_cols: List[str]) -> ColumnTransformer:
        """
        Build sklearn preprocessing pipeline for numerical features
        
        Args:
            numerical_cols: List of column names to be scaled
            
        Returns:
            Configured ColumnTransformer with StandardScaler for numerical columns
        """
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        return ColumnTransformer([
            ('num', num_pipeline, numerical_cols)
        ])

    @staticmethod
    def full_preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data preprocessing pipeline:
        1. Basic cleaning
        2. Categorical encoding
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Fully preprocessed DataFrame ready for modeling
        """
        df_clean = DataPreprocessor._basic_clean(df)
        df_encoded = DataPreprocessor._encode_categorical(df_clean)
        
        return df_encoded