import streamlit as st   #Streamlit code typically follows this import statement, where you create your web app.
import pickle   #You can use the 'pickle' module to serialize/deserialize Python objects if needed.
import string   #The 'string' module provides string constants and functions for string manipulation. You may use it for text processing.
from nltk.corpus import stopwords   #This import allows you to access the NLTK stopwords dataset. You can use it to filter out common words in text data.
import nltk    #Importing the NLTK library is required to use its various natural language processing functions.
from nltk.stem.porter import PorterStemmer
# This line imports the PorterStemmer class from NLTK. You can use it for stemming words in text data.

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SPAM Buster")

input_sms = st.text_area("Enter the Message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam,be aware of the message.")
    else:
        st.header("Not Spam.")
        st.header("You could give reply of the message if you want to")
