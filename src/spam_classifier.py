from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, Any, Tuple
import joblib

class SpamClassifier:
    """SVM-based spam classifier."""
    
    def __init__(self, **kwargs):
        """
        Initialize the spam classifier.
        
        Args:
            **kwargs: Arguments to pass to SVC
        """
        self.model = SVC(kernel='linear', probability=True, **kwargs)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
        """
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted labels
        """
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for predictions.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Probability estimates for each class
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SpamClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            SpamClassifier: Loaded classifier instance
        """
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance