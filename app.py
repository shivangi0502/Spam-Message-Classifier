import streamlit as st
import joblib
from src.preprocess import clean_text
import nltk
import os

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(app_dir, 'models')

try:
    # --- CHANGE THIS LINE ---
    # Load the model named 'lr_tfidf_model.pkl' (which likely contains the SMOTE-trained model)
    model = joblib.load(os.path.join(models_dir, 'lr_tfidf_model.pkl')) # Changed filename here!

    # Load its corresponding vectorizer
    vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
except FileNotFoundError:
    # Update the error message to reflect the new expected filename for clarity
    st.error("Model or vectorizer files not found. Make sure 'models' folder contains 'lr_tfidf_model.pkl' and 'tfidf_vectorizer.pkl'.")
    st.stop()


st.title("Spam Email Classifier")
st.write("Enter a message below to check if it's spam or ham.")

# Text input from user
user_message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_message:
        # Preprocess the user input
        cleaned_message = clean_text(user_message)

        # Transform the cleaned message using the loaded vectorizer
        message_vectorized = vectorizer.transform([cleaned_message])

        # Make prediction
        prediction = model.predict(message_vectorized)[0]
        prediction_proba = model.predict_proba(message_vectorized)[0]

        st.write("---")
        if prediction == 1:
            st.error(f"Prediction: SPAM")
            st.write(f"Confidence (Spam): {prediction_proba[1]*100:.2f}%")
            st.write(f"Confidence (Ham): {prediction_proba[0]*100:.2f}%")
        else:
            st.success(f"Prediction: HAM")
            st.write(f"Confidence (Ham): {prediction_proba[0]*100:.2f}%")
            st.write(f"Confidence (Spam): {prediction_proba[1]*100:.2f}%")
    else:
        st.warning("Please enter a message to predict.")

st.markdown("---")
st.markdown("Developed by shivangi")