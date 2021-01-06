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
st.set_option('deprecation.showPyplotGlobalUse', False)
nlp = en_core_web_sm.load()

# read in twitter data
df2020 = pd.read_csv('2020_cleaned.csv')
df2021 = pd.read_csv('2021_cleaned.csv')

# Declare functions

def keyword_search(df, keyword):
    result = df[['source','text']][df['text'].str.contains(keyword)]
    font_color = ['rgb(40,40,40)', ['rgb(255,0,0)' if word == keyword else 'rgb(10,10,10)' for word in result.text]]

    fig = go.Figure(data=[go.Table(
    header=dict(values=list(result.columns),
    fill_color='blue',
    align='center'),
    cells=dict(values=[result.source, result.text],
    fill_color='lavender',font = dict(color=font_color),
    align='left'))
    ])
    statement = f'No mentions of {keyword} in these tweets'.format(keyword)

    return statement if result.empty else fig

def find_hashtags(content):
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', content)

def count_hashtags(hashtags):
    return len(hashtags)


def sentiment_analysis(df,option):

    fig = px.scatter(df, x="datetime", # date on the x axis
               y="sentiment_polarity", # sentiment on the y axis
               hover_data=["source", "preview"], # data to show on hover
               color_discrete_map={"positive": "lightseagreen" ,
               "negative": "indianred" }, # colors to use
               color="sentiment", # what should the color depend on?
               labels={"sentiment_polarity": "Comment positivity",
                "datetime": "Date comment was posted"}, # axis names
               title=f"Sentiment in {option} resolution tweets".format(option=option), # title of figure
          )
    return(fig)


def wordcloud_generator(df, my_additional_stop_words, max_words):
    stopwords = set(STOPWORDS)
    stopwords.update(my_additional_stop_words)


    long_string = ",".join(str(v) for v in list(df['clean_tweet'].values))

    wordcloud = WordCloud(background_color="white",
    max_words=max_words,
    contour_width=3,
    contour_color='steelblue',
    stopwords=stopwords)


    plt.imshow(wordcloud.generate(long_string))
    plt.axis('off')
    plt.show()
    st.pyplot()
    return " "



def get_top_n_words(corpus, stop_words, n=None):
    vec = CountVectorizer(stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus.values.astype('U'))
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda  x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_trigram(corpus, n):
    my_additional_stop_words = ['new','year',
    'resolution', 'rewards','referral','points', 'using', 'free', 'earn',
    'straighttalkrewards', "11", "month", 'join', 'code']
    stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    corpus  = corpus.values.astype('U')

    vec = CountVectorizer(ngram_range=(3,3), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    trigram = pd.DataFrame(words_freq[:n], columns = ['trigram','count'])


    fig = px.bar(trigram, x="count", y="trigram",  orientation='h',
      title=f"Trigrams",
      labels={"trigram":"Trigram",
              "counts":"Count"},
       color_discrete_sequence=["blueviolet"],
       height=500,
       width=800)
    fig.update_xaxes(tickangle=90)

    return fig


# App body
st.title("Did Covid affect our New Year Resolutions?")
st.text('What were people saying in relation to certain words?')
option = st.selectbox(label='Select a Year', options=['','2020','2021'], key='1')
keyword = st.text_input("Type in a keyword", key='1')

if keyword:
    if option == '2020':
        st.write(keyword_search(df2020, keyword))
    else:
        st.write(keyword_search(df2021, keyword))


# Sentiment Analysis
st.title("Sentiment Analysis")
st.text('Did the sentiments change?')
option = st.selectbox(label='Select a Year', options=['','2020','2021'], key='sentiment analysis')

if option:
    if option == '2020':
        st.write(sentiment_analysis(df2020, option))
    else:
        st.write(sentiment_analysis(df2021, option))


# Word Cloud
st.title("Word Cloud")
st.text('Did the general topic theme change?')
option = st.selectbox(label='Select a Year', options=['','2020','2021'],
key='wordcloud')

additional_stop_words = st.text_input("Type words to be removed from wordclound using a comma to seperate, hint:'resolution', 'new', 'year'",
 key='wordcloud_stopwords')
max_words = st.number_input('Choose the maximum number of words to be displayed',
value=0, key='wordcloud' )


collect_words = [word.strip() for word in additional_stop_words.split(',')]

if max_words:
    if option == '2020':
        st.write(wordcloud_generator(df2020, collect_words, max_words))
    else:
        st.write(wordcloud_generator(df2021, collect_words, max_words))


st.title("Trigrams")
st.text('Did the phrases change?')
option = st.selectbox(label='Select a Year', options=['','2020','2021'], key='trigrams')

max_number = st.number_input('Choose the maximum number of trigrams to be displayed',
value=0, key='trigram_maxnumber' )

if max_number:
    if option == '2020':
        st.write(get_top_n_trigram(df2020.clean_tweet, max_number))

    else:
        st.write(get_top_n_trigram(df2021.clean_tweet, max_number))
