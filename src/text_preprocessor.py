from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple

class TextPreprocessor:
    """Class for preprocessing text data for spam classification."""
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize text preprocessor.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.scaler = StandardScaler(with_mean=False)  # Sparse matrix doesn't support centering
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the preprocessor and transform the texts.
        
        Args:
            texts (List[str]): List of text messages
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        # Convert texts to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_tfidf)
        
        return X_scaled
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using fitted preprocessor.
        
        Args:
            texts (List[str]): List of text messages
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        # Transform texts to TF-IDF features
        X_tfidf = self.vectorizer.transform(texts)
        
        # Scale the features
        X_scaled = self.scaler.transform(X_tfidf)
        
        return X_scaled