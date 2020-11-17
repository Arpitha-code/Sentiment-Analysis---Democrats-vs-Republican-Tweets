
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:56:06 2020

@author: Arpitha Jagadish
"""

import pandas as pd
import numpy as np
import os
import re
import seaborn as sns

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

import string
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import pickle
import io 
import joblib
import spacy
from wordcloud import STOPWORDS
import streamlit as st


def main():   
    def loaddata(Text):
            df = pd.read_csv("ExtractedTweets.csv")
            df.dropna(axis = 0, inplace = True)
            df.head()

            df["Party_log"] = [1 if each == "Democrat" else 0 for each in df.Party]
            print(df.shape)
            df.head()
            def lemmatize(data_str):
                # expects a string
                list_pos = 0
                cleaned_str = ''
                lmtzr = WordNetLemmatizer() 
                text = data_str.split() 
                tagged_words = pos_tag(text) 
                for word in tagged_words:
                    if 'v' in word[1].lower():
                        lemma = lmtzr.lemmatize(word[0], pos='v')
                    else:
                        lemma = lmtzr.lemmatize(word[0], pos='n')
                    if list_pos == 0: 
                        cleaned_str = lemma
                    else:
                        cleaned_str = cleaned_str + ' ' + lemma
                    list_pos += 1 
                return cleaned_str
            def remove_features(data_str): # compile regex
                url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?') 
                punc_re = re.compile('[%s]' % re.escape(string.punctuation)) 
                num_re = re.compile('(\\d+)')
                mention_re = re.compile('@(\w+)')
                alpha_num_re = re.compile("^[a-z0-9_.]+$")
                # convert to lowercase
                data_str = data_str.lower()
                # remove hyperlinks
                data_str = url_re.sub(' ', data_str)
                # remove @mentions
                data_str = mention_re.sub(' ', data_str)
                # remove puncuation
                data_str = punc_re.sub(' ', data_str)
                # remove numeric 'words'
                data_str = num_re.sub(' ', data_str)
                # remove non a-z 0-9 characters and words shorter than 1 characters 
                list_pos = 0
                cleaned_str = ''
                for word in data_str.split():
                    if list_pos == 0:
                        if alpha_num_re.match(word) and len(word) > 1:
                            cleaned_str = word 
                        else:
                            cleaned_str = ' '
                    else:
                        if alpha_num_re.match(word) and len(word) > 1:
                            cleaned_str = cleaned_str + ' ' + word 
                        else:
                            cleaned_str += ' '
                    list_pos += 1
                
                return " ".join(cleaned_str.split())
            # Cleaning Data
            data_clean = []
            for i in range(len(df.Tweet)):
                res = remove_features(df.Tweet[i])
                res1 = lemmatize(res)
                data_clean.append(res1)
                
            df['clean_headline'] = data_clean
            print(df)

        
            #add some unnecessary words to STOPWORDS list
            STOPWORDS.add("rt")
            STOPWORDS.add("s")
            STOPWORDS.add("u")
            STOPWORDS.add("amp")
            STOPWORDS.add("th")
            STOPWORDS.add("will")
            STOPWORDS.add("t")
            STOPWORDS.add("m")
            
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(df, test_size=0.3, train_size=0.7, random_state=14)
            train.shape, test.shape
            from nltk.stem.lancaster import LancasterStemmer
            st = LancasterStemmer()
            
            def token(text):
                txt = nltk.word_tokenize(text.lower())
                return [st.stem(word) for word in txt]
            
            cv = CountVectorizer(tokenizer=token,stop_words=STOPWORDS,
                                 analyzer=u'word', min_df=4)
            X_train_cv = cv.fit_transform(train['clean_headline'].tolist()) # fit_transform learns the vocab and one-hot encodes
            X_test_cv = cv.transform(test['clean_headline'].tolist()) # transform uses the same vocab and one-hot encodes
            # print the dimensions of the training set (text messages, terms)
            print(X_train_cv.shape)
            
            
            from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report
            rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            rfc.fit(X = X_train_cv, y = train['Party_log'])
            
            y_pred_rfc=rfc.predict(X_test_cv)
            print(accuracy_score(y_pred_rfc, test['Party_log']))

            vec_text=cv.transform(Text).toarray()
            result=rfc.predict(vec_text)
            print(result)
            if result==1:
                 return "demo"
            elif result==0:
                 return "rep"
                


    st.title("sentiment Analysis")
    activities=["Prediction","NLP"]
    choice=st.sidebar.selectbox("Choose actity",activities)
    if choice == "Prediction":
        Tweet_text = st.text_area("Enter Text","Type Here")
        all_ml_models = ["MNB"]
        model_choice=st.selectbox("Choose ML Model",all_ml_models)
        prediction_labels= {'Democrat':1,'Republican':0}
        if st.button("Classify"):
            st.text("Original test ::\n{}".format(Tweet_text))
            if model_choice == "MNB":
                prediction=loaddata([Tweet_text])
                if prediction == "demo":
                    Par="Democrat"
                    st.success('Party:{}'.format(Par))
                else:
                    Par="Republican"
                    st.success('Party:{}'.format(Par))
                    
                

     
 


if __name__=='__main__':
    main()


    
    
        
