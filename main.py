import streamlit as st
import pandas as pd
import joblib as jb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')

st.set_page_config(layout='wide',page_title='Flipkart Review',page_icon='🛒')

vectorizer = TfidfVectorizer() 

# Load the trained model
model = jb.load('logistic_bagging_model.joblib')

# Load the dataset
data = pd.read_csv(r'C:\Users\John Tlaletso Diale\Desktop\FlipKart_NLP\flipkart-product-customer-reviews-dataset\Dataset-SA-clean.csv')

data['preprocessed_text'] = data['preprocessed_text'].fillna('')

X_train, X_test, y_train,y_test = train_test_split(data['preprocessed_text'],data['Sentiment'],test_size=0.2,random_state=42,stratify=data['Sentiment'])

vectorizer.fit_transform(X_train)
st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
st.markdown("<h1 style='text-align: center; color: white;'>Flip Kart Review</h1>",unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white;'> 🛒 </h1>",unsafe_allow_html=True)
st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Most reviewed product
most_reviewed = data['product_name'].value_counts().index[0]
st.write(f'Most reviewed product: {most_reviewed}')
st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Create a form for user input
product = st.selectbox('Select a product', data['product_name'].unique())

# Group the reviews by product and sentiment
grouped = data.groupby(['product_name', 'Sentiment']).size().reset_index(name='count')

# Calculate the total number of reviews for the selected product
selected_totals = grouped[grouped['product_name'] == product].groupby('product_name')['count'].sum().reset_index(name='total')

# Merge the two dataframes
merged = pd.merge(grouped, selected_totals, on='product_name')

# Calculate the percentage of positive and negative reviews for the selected product
positive_count = merged.loc[(merged['product_name'] == product) & (merged['Sentiment'] == 'positive'), 'count'].values
negative_count = merged.loc[(merged['product_name'] == product) & (merged['Sentiment'] == 'negative'), 'count'].values
if len(positive_count) > 0 and len(negative_count) > 0:
    positive_percent = (positive_count / (positive_count + negative_count)) * 100
    negative_percent = 100 - positive_percent
    total_reviews = merged.loc[merged['product_name'] == product, 'total'].values[0]
    st.write(f'Percentage of positive reviews for {product}: {positive_percent[0]:.2f}%')
    st.write(f'Percentage of negative reviews for {product}: {negative_percent[0]:.2f}%')
    st.write(f'Total number of reviews for {product}: {total_reviews}')
else:
    st.write(f'There are no reviews for {product}.')
st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

review = st.text_input('Enter a review')

# Define a prediction function
def predict_review(review):
    # Preprocess the text
    review = preprocess(review)
    # Make a prediction using the model
    prediction = model.predict(review.reshape(1, -1))[0]
    return prediction

# Define a preprocessing function
def preprocess(text):
    
    text = [text] # wrap text in a list to make it an iterable of raw text documents
    text = vectorizer.transform(text)

    return text

# Make a prediction and display the result
if st.button('Predict'):
    prediction = predict_review(review)
    if prediction == 0:
        st.write('This is a bad review. 😞')
    else:
        st.write('This is a good review! 😃')
