# Spam Email Classification System

This project implements a machine learning-based spam email classification system using Python. It features a Support Vector Machine (SVM) classifier with TF-IDF vectorization for text processing.

## Features

- Email text preprocessing and vectorization using TF-IDF
- Support Vector Machine (SVM) classifier
- Interactive web interface using Streamlit
- Statistical visualizations of model performance
- Real-time email classification

## Requirements

- Python 3.11+
- scikit-learn
- pandas
- streamlit
- matplotlib
- seaborn
- plotly

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
```
3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```
2. Open your web browser and navigate to http://localhost:8501

## Project Structure

```
├── dataset/              # Training and test datasets
├── models/              # Saved model files
│   ├── spam_classifier.joblib
│   └── text_preprocessor.joblib
├── src/                 # Source code
│   ├── app.py          # Streamlit web interface
│   ├── data_loader.py  # Data loading utilities
│   ├── spam_classifier.py # SVM classifier implementation
│   └── text_preprocessor.py # Text preprocessing utilities
├── tests/              # Test files
└── requirements.txt    # Project dependencies
```

## Model Performance

The SVM classifier achieves:
- High accuracy in spam detection
- Low false positive rate
- Efficient real-time classification

Visual performance metrics and statistical analysis are available in the web interface.

## Contributors

- Sandra Lang

## License

This project is licensed under the MIT License - see the LICENSE file for details.