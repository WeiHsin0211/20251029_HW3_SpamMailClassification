import pandas as pd
import requests
from typing import Tuple
import numpy as np

class DataLoader:
    """Class for loading and preprocessing spam classification data."""
    
    def __init__(self, url: str):
        """
        Initialize DataLoader with data source URL.
        
        Args:
            url (str): URL of the CSV data source
        """
        self.url = url
        self.data = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download data from the specified URL.
        
        Returns:
            pd.DataFrame: Downloaded data
        """
        response = requests.get(self.url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save data to a temporary file and read with pandas
        with open('temp_spam_data.csv', 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        self.data = pd.read_csv('temp_spam_data.csv', names=['label', 'message'])
        return self.data
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the loaded data.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features and labels
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call download_data() first.")
            
        # Convert spam/ham labels to binary
        y = (self.data['label'] == 'spam').astype(int)
        
        # Return messages and labels
        return self.data['message'], y
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X, y = self.preprocess_data()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)