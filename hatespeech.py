# File 2: Text Preprocessing and Data Preparation for Hate Speech Detection

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

# Download NLTK stopwords
nltk.download('stopwords')

# Define text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.strip()  # Strip leading/trailing whitespace
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(stemmed_tokens)
    return cleaned_text

# Load CSV data
data = pd.read_csv('D:\\PycharmProjectszip\\PycharmProjects\\HateSpeech\\Training\\twitter_data (1).csv')

# Map class labels
data['labels'] = data['class'].map({
    0: 'Hate Speech',
    1: 'Offensive Speech',
    2: 'No Hate and Offensive Speech'
})

# Select relevant columns and apply text cleaning
data['cleaned_tweet'] = data['tweet'].apply(clean_text)

# Prepare features (x) and labels (y)
x = np.array(data['cleaned_tweet'])
y = np.array(data['labels'])
