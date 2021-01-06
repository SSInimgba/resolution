import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random
import nltk
import spacy
import en_core_web_sm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (CountVectorizer, ENGLISH_STOP_WORDS)
from sklearn.decomposition import (LatentDirichletAllocation, NMF)
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
import re
import sys
import textblob
import pyLDAvis.gensim
import swifter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
nlp = en_core_web_sm.load()

# read in twitter data
df2020 = pd.read_csv('2020.csv')
df2021 = pd.read_csv('2021.csv')

# Declare functions


def remove_links(tweet):
    # remove http links
    tweet = re.sub('http\S+', '', tweet)
    # remove bitly links
    tweet = re.sub('bit.ly/\S+', '', tweet)
    # remove bitly links
    tweet = tweet.strip('[link]')
    return tweet

def remove_users(tweet):
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('@[^\s]+', '', tweet) # remove tweeted at
    return tweet


# cleaning master function
def clean_tweet(tweet):
    my_stopwords = nltk.corpus.stopwords.words('english')
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords
    return tweet

def lemmatizer(text):
    sentence = []
    doc = nlp(text)

    for word in doc:
        sentence.append(word.lemma_)
    return " ".join(sentence)


def apply_clean_tweets(df):
    df['clean_text'] = df.text.swifter.apply(clean_tweet)
    df['clean_text'] = df.clean_text.swifter.apply(lemmatizer)
