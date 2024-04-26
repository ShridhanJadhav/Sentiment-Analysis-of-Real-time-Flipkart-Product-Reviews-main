import streamlit as st
import re
import os
#import emojiclear

#import autocorrect ## Has to be installed
#import nltk
#from nltk.tokenize import word_tokenize,sent_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer,WordNetLemmatizer,LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,ConfusionMatrixDisplay
import pickle
result = None


# import os
# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct the path to the pickle file relative to the current script
pickle_file_path = os.path.join(current_dir, "senti.pkl")

# Load the pickle file
with open(pickle_file_path, "rb") as f:
    model = pickle.load(f)

st.title("Sentiment Analysis App")
text = st.text_input("Enter your review below:")



if st.button("Submit")==True:
    result = model.predict([text])[0]
    st.write(result)

if result == 'Positive':
    st.write("This is a positive review!")
elif result == 'Negative':
    st.write("This is a negative review!")