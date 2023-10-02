from flask import Flask, render_template, request
import pandas as pd
import joblib
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the pre-trained sentiment analysis model
model = joblib.load("sentiment_model.pkl")

# Load the CountVectorizer
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
        return render_template('result.html', text=text, sentiment=sentiment)

def predict_sentiment(text):
    # Vectorize the input text
    text_vector = vectorizer.transform([text])

    # Predict sentiment using the pre-trained model
    sentiment = model.predict(text_vector)[0]
    return sentiment

if __name__ == '__main__':
    app.run(debug=True)
