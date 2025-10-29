import streamlit as st
import pandas as pd
from data_loader import DataLoader
from text_preprocessor import TextPreprocessor
from spam_classifier import SpamClassifier
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

def create_confusion_matrix_plot(y_true, y_pred):
    """Create and return a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig

def create_roc_curve_plot(y_true, y_prob):
    """Create and return an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                            name=f'ROC curve (AUC = {roc_auc:.2f})',
                            mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            name='Random guess',
                            mode='lines',
                            line=dict(dash='dash')))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600
    )
    return fig

def create_feature_importance_plot(classifier, preprocessor, n_features=20):
    """Create and return a feature importance plot."""
    if hasattr(classifier.model, 'coef_'):
        # Get feature names and their coefficients
        feature_names = preprocessor.vectorizer.get_feature_names_out()
        coefficients = classifier.model.coef_[0]
        
        # Convert sparse matrix to dense if necessary
        if hasattr(coefficients, 'toarray'):
            coefficients = coefficients.toarray().flatten()
            
        # Calculate absolute values
        importance_values = np.abs(coefficients)
        
        # Create DataFrame of features and their importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        })
        # Sort and select top N features
        feature_importance = feature_importance.nlargest(n_features, 'importance')
        
        # Create bar plot using plotly
        fig = px.bar(feature_importance, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title=f'Top {n_features} Most Important Words')
        
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        return fig
    return None

def load_or_train_model():
    """Load existing model or train a new one if it doesn't exist."""
    model_path = 'models/spam_classifier.joblib'
    preprocessor_path = 'models/text_preprocessor.joblib'
    
    # Store evaluation metrics in session state
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    
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
            X_test_processed = preprocessor.transform(X_test)
            
            # Train model
            classifier.train(X_train_processed, y_train)
            
            # Save model and preprocessor
            if not os.path.exists('models'):
                os.makedirs('models')
            classifier.save_model(model_path)
            preprocessor.save_preprocessor(preprocessor_path)
            
            # Calculate and store evaluation metrics
            y_pred = classifier.predict(X_test_processed)
            y_prob = classifier.predict_proba(X_test_processed)[:, 1]
            
            st.session_state.model_metrics = {
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
    return classifier, preprocessor

def main():
    st.title('📧 Spam Email Classifier')
    st.write("""
    This application uses machine learning to classify emails as spam or not spam.
    Enter your email text below to check if it's spam!
    """)
    
    # Add a navigation menu
    page = st.sidebar.selectbox(
        "Navigation",
        ["Classify Email", "Model Performance", "Feature Analysis"]
    )
    
    # Load or train the model
    classifier, preprocessor = load_or_train_model()

    if page == "Classify Email":
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

    elif page == "Model Performance":
        st.write("### Model Performance Metrics")
        
        if st.session_state.model_metrics is not None:
            metrics = st.session_state.model_metrics
            
            # Display confusion matrix
            st.write("#### Confusion Matrix")
            st.write("This matrix shows how well the model classifies both spam and non-spam messages:")
            cm_fig = create_confusion_matrix_plot(metrics['y_test'], metrics['y_pred'])
            st.pyplot(cm_fig)
            
            # Display ROC curve
            st.write("#### ROC Curve")
            st.write("This curve shows the trade-off between true positive rate and false positive rate:")
            roc_fig = create_roc_curve_plot(metrics['y_test'], metrics['y_prob'])
            st.plotly_chart(roc_fig, use_container_width=True)
            
            # Calculate and display additional metrics
            accuracy = np.mean(metrics['y_test'] == metrics['y_pred'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                precision = np.sum((metrics['y_pred'] == 1) & (metrics['y_test'] == 1)) / np.sum(metrics['y_pred'] == 1)
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                recall = np.sum((metrics['y_pred'] == 1) & (metrics['y_test'] == 1)) / np.sum(metrics['y_test'] == 1)
                st.metric("Recall", f"{recall:.2%}")
        else:
            st.warning("Model metrics are not available. Please retrain the model.")
            
    elif page == "Feature Analysis":
        st.write("### Feature Importance Analysis")
        st.write("This visualization shows the most influential words in determining whether a message is spam:")
        
        feature_importance_fig = create_feature_importance_plot(classifier, preprocessor)
        if feature_importance_fig is not None:
            st.plotly_chart(feature_importance_fig, use_container_width=True)
        
        st.write("""
        #### Understanding Feature Importance
        - Positive values indicate words more associated with spam
        - The larger the absolute value, the stronger the association
        - These words are the most influential in the model's decision-making process
        """)

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