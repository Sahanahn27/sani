
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os  # Import the os module

# Load dataset directly
def load_data():
    """
    Loads data from an uploaded file or uses a default file if none is uploaded.

    Returns:
        pd.DataFrame: The loaded data, or None if loading failed.
    """
    
    # Display a file uploader widget
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
    
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Try to read the file using pandas
        try:
            data = pd.read_excel(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None  # Or handle the error differently
    else:
        st.warning("Please upload an Excel file.")
        return None


# Preprocess the data
def preprocess_data(data):
    # Assuming you have already cleaned the data in your notebook
    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=['voice.plan', 'intl.plan'], drop_first=True)

    # Drop any unnecessary columns
    data.drop(columns=['state', 'area.code', 'intl.charge', 'eve.charge', 'night.charge', 'day.charge'], inplace=True)

    # Define features and target
    X = data.drop('churn', axis=1)
    y = data['churn'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert churn column to binary (1 for churn, 0 for no churn)
    return X, y

# Train the model directly in the app
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Optionally, display model performance on test set
    y_pred = model.predict(X_test)
    st.write("Model Performance on Test Set:")
    st.text(classification_report(y_test, y_pred))

    return model

# Main app code
st.title("Customer Churn Prediction App")

# Load and preprocess data
data = load_data()  # Call the modified load_data function

# Check if data was loaded successfully
if data is not None:
    X, y = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # Collect user input for prediction
    # ... (rest of your code) ...
