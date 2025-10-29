from data_loader import DataLoader
from text_preprocessor import TextPreprocessor
from spam_classifier import SpamClassifier
import os

def main():
    # Data source URL
    DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    
    # Initialize components
    data_loader = DataLoader(DATA_URL)
    preprocessor = TextPreprocessor(max_features=1000)
    classifier = SpamClassifier(random_state=42)
    
    # Load and split data
    print("Loading data...")
    data_loader.download_data()
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split()
    
    # Preprocess data
    print("Preprocessing text data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train model
    print("Training spam classifier...")
    classifier.train(X_train_processed, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = classifier.evaluate(X_test_processed, y_test)
    
    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    classifier.save_model('models/spam_classifier.joblib')
    print("\nModel saved to models/spam_classifier.joblib")

if __name__ == "__main__":
    main()