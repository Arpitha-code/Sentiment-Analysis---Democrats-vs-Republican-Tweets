
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:56:06 2020

@author: Arpitha Jagadish
"""

import pandas as pd
import re
import nltk
#nltk.download("punkt")


import string
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

#import pickle
#import io 
#import joblib
import spacy

nlp = spacy.load("en_core_web_sm")

import streamlit as st
#import neattext as nt
#from neattext.functions import clean_text
from wordcloud import STOPWORDS
from nltk.stem.lancaster import LancasterStemmer


def main():
    
    def loaddata(Text,mods):
            #read the preprocessed data from pickle file
            df = pd.read_pickle("corpus.pkl")
            
            STOPWORDS.add("rt")
            STOPWORDS.add("s")
            STOPWORDS.add("u")
            STOPWORDS.add("amp")
            STOPWORDS.add("th")
            STOPWORDS.add("will")
            STOPWORDS.add("t")
            STOPWORDS.add("m")
            STOPWORDS.add("today")
           
            
            #split the data into train and test set
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(df, test_size=0.3, train_size=0.7, random_state=14)
            
            
            #performing stemming
            lt = LancasterStemmer()
            def token(text):
                txt = nltk.word_tokenize(text.lower())
                return [lt.stem(word) for word in txt]
            
            #document term matrix using Tfidf vectorizer
            tfv = TfidfVectorizer(tokenizer=token,stop_words=STOPWORDS,analyzer=u'word', min_df=4)
            X_train_tfv = tfv.fit_transform(train['clean_tweet']) 
            X_test_tfv = tfv.transform(test['clean_tweet']) 
        
           
            X_train_tfv = pd.DataFrame(X_train_tfv.toarray(), columns=tfv.get_feature_names())
            X_test_tfv = pd.DataFrame(X_test_tfv.toarray(), columns=tfv.get_feature_names())
            
            if(mods=="MNB"):
                
                st.success("Performing MNB Classification")
                #build the model
                nb = MultinomialNB()
                # Train the model
                nb.fit(X_train_tfv, train['Party_log'])
                
                #transform the entered text into document term matrix
                vec_text = tfv.transform(Text).toarray()
                #predicting the value for newly entered tweet
                result = nb.predict(vec_text)
                #if result is 1 then democrat else republican
            else:
                st.success("Performing Logistic Regression")
                #build the model
                lr = LogisticRegression()
                # Train the model
                lr.fit(X_train_tfv, train['Party_log'])
                
                #transform the entered text into document term matrix
                vec_text = tfv.transform(Text).toarray()
                #predicting the value for newly entered tweet
                result = lr.predict(vec_text)
                #if result is 1 then democrat else republican
                
            if result == 1:
                 return "demo"
            elif result == 0:
                 return "rep"
                
    
    st.title("Sentiment Analysis ")
    st.title("Democrats vs Republicans Twitter Data")
    # available NlP techniques
    activities=["Prediction","NLP"]
    
    #using streamlit sidebar option
    choice = st.sidebar.selectbox("Select Activity",activities)
    
    #if prediction is chosen 
    if choice == "Prediction":
        #read the text from the text_area
        Tweet_text = st.text_area("Enter Text","Type Here")
        
        #cleaning the tweet entered
        url_re = re.compile('http\S+') 
        punc_re = re.compile('[%s]' % re.escape(string.punctuation)) 
        num_re = re.compile('(\d+)')
        alpha_num_re = re.compile("[^a-zA-Z]")
        # convert to lowercase
        Tweet_text = Tweet_text.lower()
        # remove hyperlinks
        Tweet_text = url_re.sub(' ', Tweet_text)
        # remove puncuation
        Tweet_text = punc_re.sub(' ', Tweet_text)
        # remove numeric 'words'
        Tweet_text = num_re.sub(' ', Tweet_text)
        Tweet_text = alpha_num_re.sub(' ', Tweet_text)
                    
        #just considering the model with highest accuracy, can include other models
        all_ml_models = ["MNB","LRM"]
        #display the models using streamlit selectbox
        model_choice=st.selectbox("Choose ML Model",all_ml_models)
        
        #streamlit button
        if st.button("Classify"):
            
            #displaying the preprocessed data
            st.text("Pre-Processed Data (stop words will be removed while creating document term matrix(tfidf Vectorizer))::\n{}".format([Tweet_text]))
            
            #if statement runs depending on the model chosen 
            if model_choice == "MNB":
                
                st.success("You have chosen Multinominal Naive Bayes model")
                #function loaddata returns the predicted party
                prediction = loaddata([Tweet_text],model_choice)
                
                
                if prediction == "demo":
                    #display the results
                    
                    st.success('Party:{}'.format("Democrat"))
                    #path for the image
                    image='Images/Democrat.jpg'
                    img=Image.open(image)
                    #display the image 
                    st.image(img,width=300)
                    
                else:
                    st.success('Party:{}'.format("Republican"))
                    image='Images/Republican.jpg'
                    img=Image.open(image)
                    st.image(img,width=300)
                    
            if model_choice == "LRM":
                
                st.success("You have chosen Logistic Regression model")
                #function loaddata returns the predicted party
                prediction = loaddata([Tweet_text],model_choice)
                
                
                if prediction == "demo":
                    #display the results
                    st.success('Party:{}'.format("Democrat"))
                    #path for the image
                    image='Images/Democrat.jpg'
                    img=Image.open(image)
                    #display the image 
                    st.image(img,width=300)
                else:
                    st.success('Party:{}'.format("Republican"))
                    image='Images/Republican.jpg'
                    img=Image.open(image)
                    st.image(img,width=300)
                    
                    
    
    #if chosen option is nlp
    if choice == 'NLP':
        st.info("Natural Language Processing")
        
        #enter the new tweet
        Tweet_text = st.text_area("Enter Here","Type Here")
        
        #cleaning the tweet entered
        url_re = re.compile('http\S+') 
        punc_re = re.compile('[%s]' % re.escape(string.punctuation)) 
        num_re = re.compile('(\\d+)')
        alpha_num_re = re.compile("[^a-zA-Z]")
        # convert to lowercase
        Tweet_text = Tweet_text.lower()
        # remove hyperlinks
        Tweet_text = url_re.sub(' ', Tweet_text)
        # remove puncuation
        Tweet_text = punc_re.sub(' ', Tweet_text)
        # remove numeric 'words'
        Tweet_text = num_re.sub(' ', Tweet_text)
        Tweet_text = alpha_num_re.sub(' ', Tweet_text)
        
        STOPWORDS.add("rt")
        STOPWORDS.add("s")
        STOPWORDS.add("u")
        STOPWORDS.add("amp")
        STOPWORDS.add("th")
        STOPWORDS.add("will")
        STOPWORDS.add("t")
        STOPWORDS.add("m")
        
        
       
        list_pos = 0
        cleaned_str = ''
        text = Tweet_text.split()
        for word in text:
            if word not in STOPWORDS:
                if list_pos == 0:
                    cleaned_str = word
                else:
                    cleaned_str = cleaned_str + ' ' + word
                list_pos += 1
            
        
        #clean tweet             
        Tweet_text = cleaned_str
        #optoin available 
        nlp_options=["Tokenization","Lemmatization","POS Tags"]
        #selected option 
        nlp_choice=st.selectbox("Choose the NlP option",nlp_options)
        
        if st.button("Start"):
            
            #displaying the cleaned tweet
            st.info("Original Text::\n{}".format(Tweet_text))
            
            #nlp coverts text to processed docx object that is understood by spacy
            Sentence= nlp(Tweet_text)
            if nlp_choice=='Tokenization':
                result=[token.text for token in Sentence]
            elif nlp_choice == 'Lemmatization':
                result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in Sentence]
            elif nlp_choice == 'POS Tags':
                result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in Sentence]

            st.json(result)
            
            #display the results in table form
        if st.button("Tabulize"):
            docx = nlp(Tweet_text)
            c_tokens = [token.text for token in docx ]
            c_lemma = [token.lemma_ for token in docx ]
            c_pos = [token.pos_ for token in docx ]
            
            #creating dataframe using the results
            new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
            #display df
            st.dataframe(new_df)
        
        #display using wordcloud
        if st.checkbox("WordCloud"):
            fig, ax = plt.subplots(figsize=(15,5))
            c_text = Tweet_text
            wordcloud = WordCloud().generate(c_text)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig)
            st.set_option('deprecation.showPyplotGlobalUse', False)

                   
if __name__=='__main__':
    main()


    
    
        
