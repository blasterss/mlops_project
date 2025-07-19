import pandas as pd
from ..utils.config import Config

class DataLoader:
    """
    A utility class for loading and splitting Titanic dataset files.
    
    Provides static methods for:
    - Loading training/test data
    - Loading submission template
    - Splitting data into train/test sets
    """
    
    @staticmethod
    def load_data(is_train: bool = True) -> pd.DataFrame:
        """
        Load Titanic dataset from configured paths
        
        Args:
            is_train (bool): Whether to load training data (True) or test data (False)
            
        Returns:
            pd.DataFrame: Loaded dataset as a pandas DataFrame
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        try:
            file_path = Config.TRAIN_DATA if is_train else Config.TEST_DATA
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found at {file_path}") from e
        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError(f"Empty file at {file_path}") from e
    
    @staticmethod
    def load_sub_data() -> pd.DataFrame:
        """
        Load submission file template
        
        Returns:
            pd.DataFrame: Submission template with PassengerId column
            
        Raises:
            FileNotFoundError: If the submission template doesn't exist
        """
        try:
            return pd.read_csv(Config.SUBMISSION_SOURCE)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Submission template not found at {Config.SUBMISSION_SOURCE}"
            ) from e
    
    @staticmethod
    def split_data(
        df: pd.DataFrame, 
        target: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple:
        """
        Split dataset into training and testing sets
        
        Args:
            df (pd.DataFrame): Input DataFrame containing both features and target
            target (str): Name of the target column
            test_size (float): Proportion of dataset to include in test split (0-1)
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) - Split features and targets
            
        Example:
            >>> X_train, X_test, y_train, y_test = DataLoader.split_data(df, 'Survived')
        """
        from sklearn.model_selection import train_test_split
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
            
        X = df.drop(columns=[target])
        y = df[target]
        
        return train_test_split(
            X, 
            y, 
            test_size=test_size, 
            random_state=random_state
        )