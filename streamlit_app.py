#streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Function to train the model
def train_model(csv_file_path):
    os.system(f'python model_training.py {csv_file_path}')

# Function to load the model and preprocessors
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, vectorizer, label_encoder

# Function to predict root cause
def predict_root_cause(model, vectorizer, label_encoder, text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    predicted_root_cause = label_encoder.inverse_transform(prediction)[0]
    return predicted_root_cause

# Main Streamlit app
def main():
    st.title('Root Cause Classification App')

    # Sidebar - File upload
    st.sidebar.title('Upload CSV File')
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Save uploaded file to disk
        with open('uploaded_file.csv', 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Train the model
        st.subheader('Training the Model')
        train_model('uploaded_file.csv')
        st.write('Model is ready to deploy!')

        # Load the trained model and preprocessors
        model, vectorizer, label_encoder = load_model()

        # Input text for prediction
        st.subheader('Enter a Customer Query')
        text_input = st.text_area('Input text:', '')

        if st.button('Predict Root Cause'):
            if text_input:
                # Predict root cause
                predicted_root_cause = predict_root_cause(model, vectorizer, label_encoder, text_input)
                st.success(f'Predicted Root Cause: {predicted_root_cause}')

if __name__ == '__main__':
    main()