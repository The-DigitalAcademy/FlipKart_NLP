import streamlit as st
import pandas as pd
import joblib as jb
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')

vectorizer = TfidfVectorizer() 

# Load the trained model
model = jb.load('logistic_bagging_model.joblib')

# Load the dataset
data = pd.read_csv(r'C:\Users\John Tlaletso Diale\Desktop\FlipKart_NLP\flipkart-product-customer-reviews-dataset\Dataset-SA-clean.csv')


vectorizer.fit_transform(data['preprocessed_text'])

# Create a form for user input
product = st.selectbox('Select a product', data['product_name'].unique())
review = st.text_input('Enter a review')

# Define a prediction function
def predict_review(review):
    # Preprocess the text
    review =vectorizer.transform(review)
    # Make a prediction using the model
    prediction = model.predict([review])[0]
    return prediction

# Define a preprocessing function
def preprocess(text):
    # Implement your preprocessing techniques here
    return text

# Make a prediction and display the result
if st.button('Predict'):
    prediction = predict_review(review)
    if prediction == 0:
        st.write('This is a bad review.')
    else:
        st.write('This is a good review.')
