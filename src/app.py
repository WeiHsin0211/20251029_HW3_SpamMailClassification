import streamlit as st
import pandas as pd
from data_loader import DataLoader
from text_preprocessor import TextPreprocessor
from spam_classifier import SpamClassifier
import os
import numpy as np

def load_or_train_model():
    """Load existing model or train a new one if it doesn't exist."""
    model_path = 'models/spam_classifier.joblib'
    preprocessor_path = 'models/text_preprocessor.joblib'
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        classifier = SpamClassifier.load_model(model_path)
        preprocessor = TextPreprocessor.load_preprocessor(preprocessor_path)
    else:
        # Initialize components
        DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
        
        with st.spinner('Training model... This may take a moment.'):
            data_loader = DataLoader(DATA_URL)
            preprocessor = TextPreprocessor(max_features=1000)
            classifier = SpamClassifier(random_state=42)
            
            # Load and split data
            data_loader.download_data()
            X_train, X_test, y_train, y_test = data_loader.get_train_test_split()
            
            # Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train)
            
            # Train model
            classifier.train(X_train_processed, y_train)
            
            # Save model and preprocessor
            if not os.path.exists('models'):
                os.makedirs('models')
            classifier.save_model(model_path)
            preprocessor.save_preprocessor(preprocessor_path)
            
    return classifier, preprocessor

def main():
    st.title('📧 Spam Email Classifier')
    st.write("""
    This application uses machine learning to classify emails as spam or not spam.
    Enter your email text below to check if it's spam!
    """)
    
    # Load or train the model
    classifier, preprocessor = load_or_train_model()
    
    # Create text input
    email_text = st.text_area("Enter email text:", height=200)
    
    if st.button('Classify'):
        if email_text:
            # Preprocess the input text
            X_processed = preprocessor.transform([email_text])
            
            # Make prediction
            prediction = classifier.predict(X_processed)[0]
            probability = classifier.predict_proba(X_processed)[0]
            
            # Display result with proper formatting
            st.write("---")
            st.write("### Results")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 Spam Detected!")
                else:
                    st.success("✅ Not Spam")
            
            with col2:
                confidence = probability[1] if prediction == 1 else probability[0]
                st.metric(label="Confidence", value=f"{confidence:.2%}")
            
            # Add explanation
            st.write("---")
            st.write("### Explanation")
            if prediction == 1:
                st.write("""
                This email was classified as spam because it shows characteristics commonly found in spam messages.
                Consider checking for:
                - Suspicious offers or promises
                - Urgency or pressure tactics
                - Poor grammar or unusual formatting
                - Requests for personal information
                """)
            else:
                st.write("""
                This email appears to be legitimate. However, always exercise caution with:
                - Unexpected attachments
                - Requests for sensitive information
                - Unusual sender addresses
                - Unexpected urgency
                """)
        else:
            st.warning("Please enter some text to classify!")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This spam classifier uses:
        - Support Vector Machine (SVM) algorithm
        - TF-IDF for text feature extraction
        - Trained on a dataset of labeled spam and non-spam messages
        
        The model evaluates various aspects of the message including:
        - Word frequency and patterns
        - Text structure
        - Common spam indicators
        
        Note: While this model is effective, it should be used as one of many tools in identifying spam.
        Always use your judgment when dealing with suspicious emails.
        """)

if __name__ == "__main__":
    main()